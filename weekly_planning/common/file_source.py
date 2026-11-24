"""file_source: get a file the user named as a local copy -- the name is a path on this PC or a SharePoint link.

    source = FileSource(tenant_id, client_id)
    source.check(text, "SU_Others")                              # cheap check before anything runs; raises SourceError
    got = source.fetch(text, folder, "01_SU_Others", (".xlsm", ".xlsx"))      # -> Fetched(path, name)

The file at the place it came from is never changed: a local file is copied, a SharePoint file is downloaded (read only; no rename,
no lock). One FileSource logs in to SharePoint once and uses that login for every file it fetches.

The SharePoint part is common/file_operation_sharepoint.py, loaded only when a link is used (it needs the libraries msal and
requests, which a program that only uses local files does not need).
"""
from __future__ import annotations

import re
import shutil
from collections import namedtuple
from pathlib import Path

LINK = re.compile(r"^https?://", re.IGNORECASE)
LOCK_PREFIX = "_"          # 02_UpdateSchedule renames SU_Others to _SU_Others while it writes the result

Fetched = namedtuple("Fetched", "path name")      # path: the local copy; name: the file name where it came from


class SourceError(Exception):
    """The file cannot be got; the message says why in words the user can act on."""


def is_link(text):
    return bool(LINK.match(str(text or "").strip()))


class FileSource:
    def __init__(self, tenant_id=None, client_id=None, sign_in=None, client_factory=None):
        """sign_in(url, code) -> bool: shows the sign-in code to the user (a window that is safe to use from the thread that fetches);
        None = the window of file_operation_sharepoint. client_factory(): a replacement for the SharePoint client (tests)."""
        self.tenant_id = (tenant_id or "").strip()
        self.client_id = (client_id or "").strip()
        self.sign_in = sign_in
        self.client_factory = client_factory
        self._client = None

    # ------------------------------------------------------------------------------------------ before the run
    def check(self, text, label):
        """Raises SourceError when `text` cannot work: no such local file, or a link without the settings or libraries for it."""
        text = str(text or "").strip()
        if not text:
            raise SourceError(f"the {label} file was not chosen")
        if not is_link(text):
            if not Path(text).is_file():
                raise SourceError(f"the {label} file was not found: {text}")
            return
        if not (self.tenant_id and self.client_id):
            raise SourceError(f"the {label} is a SharePoint link, but tenant_id and client_id are not set in the settings file")
        if self.client_factory is None:
            try:
                import msal   # noqa: F401
                import requests   # noqa: F401
            except ImportError as e:
                raise SourceError(f"the {label} is a SharePoint link, but the library {e.name!r} is missing "
                                  f"(python -m pip install msal requests)") from None

    # ------------------------------------------------------------------------------------------ the copy
    def fetch(self, text, folder, stem, suffixes=None):
        """A copy of the file in `folder` named stem + the extension of the original (lower case). `suffixes`: the extensions that
        are accepted (an error otherwise). An old file of that name in `folder` is replaced."""
        text = str(text or "").strip()
        folder = Path(folder)
        if not is_link(text):
            source = Path(text)
            dest = folder / (stem + source.suffix.lower())
            self._accept(source.name, suffixes)
            shutil.copy2(source, dest)
            return Fetched(dest, source.name)
        try:
            client = self._get_client()
            url = client.get_file_url_from_sharing_link(text)
            name = client.get_file_name(url)
            if name.startswith(LOCK_PREFIX):
                raise SourceError(f"the file on SharePoint is named {name}: it is locked while 02_UpdateSchedule writes its result "
                                  f"(or a lock was left behind). Try again later; if it stays, ask who ran 02_UpdateSchedule.")
            self._accept(name, suffixes)
            dest = folder / (stem + Path(name).suffix.lower())
            if dest.exists():
                dest.unlink()
            client.download_file(url, dest)
        except SourceError:
            raise
        except Exception as e:      # the SharePoint client's own errors already carry a message for the user
            kind = "" if type(e).__name__ == "FileOperationSharePointError" else f"{type(e).__name__}: "
            raise SourceError(f"cannot get the file from SharePoint ({kind}{e})") from e
        return Fetched(dest, name)

    @staticmethod
    def _accept(name, suffixes):
        if suffixes and Path(name).suffix.lower() not in suffixes:
            raise SourceError(f"the file must be one of {', '.join(suffixes)}: {name}")

    def _get_client(self):
        if self._client is None:
            if self.client_factory is not None:
                self._client = self.client_factory()
            else:
                from common.file_operation_sharepoint import FileOperationSharePoint
                sign_in = self.sign_in
                if sign_in is None:
                    self._client = FileOperationSharePoint(self.tenant_id, self.client_id)
                else:
                    class Client(FileOperationSharePoint):
                        @staticmethod
                        def _show_sign_in_message(verification_uri, user_code):
                            return bool(sign_in(verification_uri, user_code))

                    self._client = Client(self.tenant_id, self.client_id)
        return self._client
