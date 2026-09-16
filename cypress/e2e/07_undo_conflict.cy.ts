/**
 * Test Suite 07 — Own-Action Conflict-Aware Undo
 *
 * Verifies the design's signature guarantee from the live app's point of
 * view: a participant's own pending Undo entry gets blocked (with a visible
 * error message, and no visible change to the schedule) once a *different*
 * participant has touched the same object since.
 *
 * The unit/integration proof for this feature already lives in
 * `AppContext.test.tsx` (`applyRemoteAction` + `hasConflict` at the reducer
 * level) — that is this feature's load-bearing evidence. This spec is
 * corroborating: a real two-socket, real-server demonstration that the
 * blocked-undo message actually renders in the live UI.
 *
 * Why a raw second socket instead of `06_viewer_parity.cy.ts`'s two-role-URL
 * pattern: this app's session UI was reworked (multi-session list, no more
 * copyable edit-link field in セッション情報 — see SessionInfoDialog) since
 * that spec was written, so there's no longer a share-link string to lift a
 * session id from. `06`'s own flow (共同編集 → セッションを開始 →
 * "input[readonly]" edit link) no longer matches the real app either.
 * Re-visiting a fresh `?session=...&role=edit` URL to play a second
 * participant would also tear down the FIRST participant's tab/socket —
 * fine for "does a late joiner see synced state" (06's actual assertion),
 * useless here, where the whole point is that the FIRST participant's own
 * live UI reacts to a second participant's edit while still mounted.
 *
 * So this spec drives one real UI participant (Editor B, the actual
 * browser-rendered app) plus one real second socket.io-client connection
 * ("Ghost") opened directly against the same collab relay the app itself
 * talks to — a real second participant on the real server, just without a
 * second rendered UI. The session id and relay URL are lifted from the
 * app's own `POST /api/sessions` / `POST /api/sessions/:id/open` calls via
 * cy.intercept, and Ghost's edit is captured directly from the actual
 * `UPDATE_ASSIGNMENT` action Editor B's own edit sends over the wire (so
 * the assignment index Ghost re-targets is never guessed).
 *
 * Note on command ordering: every value Ghost needs (the socket instance,
 * the promise that resolves with Editor B's own action) is captured into a
 * plain closure variable from inside a `.then()`, never via a *nested*
 * `cy.*()` call — a `cy.*()` issued from inside another command's `.then()`
 * gets spliced into the queue immediately after that command, ahead of
 * whatever was already queued after it at the top level. A nested
 * `cy.wrap()` waiting on "Editor B's next action" would then run BEFORE the
 * type/blur commands that actually produce that action, deadlocking until
 * its timeout. Plain variables sidestep that entirely.
 *
 * Requires the real collab server: `npm run dev:all`, not just `npm run dev`.
 */
import { io, Socket } from 'socket.io-client';

const REMARKS_PLACEHOLDER = '備考を入力...';
const UNDO_BLOCKED_MESSAGE = '他の参加者がこの対象を変更したため、元に戻せません。';

describe('07 – Undo Conflict', () => {
  it('blocks Undo and shows a message when another participant touched the same object since', () => {
    let ghost: Socket;
    let editorBActionPromise: Promise<{ type: string; payload: { index: number; updates: Record<string, unknown> } }>;

    cy.intercept('POST', '**/api/sessions').as('createSession');
    cy.intercept('POST', '**/api/sessions/*/open').as('openSession');

    cy.visit('/');
    cy.loadFixtures();

    // Create an online session straight from the schedule we just loaded —
    // the current SessionDialog's 作成 flow, not the old 共同編集 one.
    cy.contains('ファイル').click();
    cy.contains('オンラインセッションを作成').click();
    cy.contains('オンラインセッションを作成').should('be.visible');
    cy.get('input[placeholder="ニックネームを入力"]').type('Editor B');
    cy.get('input[placeholder="セッション名を入力"]').type('Undo Conflict Test');
    cy.contains('button', '現在のスケジュールから作成').click();

    let sessionId: string;
    let relayUrl: string;

    cy.wait('@createSession').then(({ response }) => {
      sessionId = response!.body.sessionId as string;
    });
    cy.wait('@openSession').then(({ response }) => {
      relayUrl = response!.body.relayUrl as string;
    });

    // Editor B's own session should now be live.
    cy.contains(/人が参加中/, { timeout: 8000 }).should('exist');

    // Ghost: a second, independent edit-role socket client on the same real
    // relay — a genuine second participant, just with no rendered UI of its
    // own. Captured into the outer `ghost` variable (not a Cypress alias) so
    // later steps can use it without any nested cy.*() calls.
    cy.then(() => new Promise<void>((resolve) => {
      const g = io(relayUrl, { path: '/collab/socket.io', transports: ['websocket', 'polling'] });
      g.on('connect', () => g.emit('join', { sessionId, name: 'Editor Ghost', role: 'edit' }));
      g.once('sync-init', (payload: { ok: boolean }) => {
        if (!payload.ok) throw new Error('Ghost failed to join the session');
        ghost = g;
        resolve();
      });
    }));

    // Editor B should now see two participants.
    cy.contains(/2人が参加中/, { timeout: 8000 }).should('exist');

    // Select an assignment bar and edit its remarks as Editor B — this is
    // Editor B's own edit, captured into their myPendingUndo stack.
    cy.get('[data-testid="assignment-bar"]', { timeout: 8000 }).should('have.length.greaterThan', 0);
    cy.get('[data-testid="assignment-bar"]').first().click();
    cy.get(`textarea[placeholder="${REMARKS_PLACEHOLDER}"]`).should('be.visible');

    // Arm the listener for Editor B's own outgoing action BEFORE triggering
    // it (a plain variable assignment, not a queued cy command — the
    // `.once()` call below registers synchronously right now).
    cy.then(() => {
      editorBActionPromise = new Promise((resolve) => ghost.once('action', resolve));
    });

    cy.get(`textarea[placeholder="${REMARKS_PLACEHOLDER}"]`).clear().type('Editor B edit');
    cy.get(`textarea[placeholder="${REMARKS_PLACEHOLDER}"]`).blur();

    // Ghost now "conflicts" Editor B by editing the exact same assignment's
    // same field to a different value, using the real index Editor B's own
    // client just sent — never guessed.
    cy.then({ timeout: 10000 }, () => editorBActionPromise).then((editorBAction) => {
      const { index } = editorBAction.payload;
      ghost.emit('action', {
        type: 'UPDATE_ASSIGNMENT',
        payload: { index, updates: { description: 'Ghost overwrote this' } },
      });
    });

    // Deselect and reselect the same bar to force the side panel's local
    // draft to resync from the now-updated assignment (the remarks textarea
    // only resyncs its draft when the *selection* changes, not on every prop
    // change) — proves Ghost's remote edit actually reached Editor B's live
    // client.
    cy.get('[data-testid="assignment-bar"]').first().click();
    cy.get('[data-testid="assignment-bar"]').first().click();
    cy.get(`textarea[placeholder="${REMARKS_PLACEHOLDER}"]`).should('have.value', 'Ghost overwrote this');

    // Editor B tries to undo their own remarks edit — it should be blocked,
    // because Ghost touched the same assignment since.
    cy.contains('button', '元に戻す').should('not.be.disabled').click();
    cy.contains(UNDO_BLOCKED_MESSAGE, { timeout: 8000 }).should('be.visible');

    // Nothing changed: still Ghost's value, not reverted to Editor B's edit
    // or to the original empty remarks.
    cy.contains('button', '閉じる').click();
    cy.get('[data-testid="assignment-bar"]').first().click();
    cy.get('[data-testid="assignment-bar"]').first().click();
    cy.get(`textarea[placeholder="${REMARKS_PLACEHOLDER}"]`).should('have.value', 'Ghost overwrote this');

    cy.then(() => ghost.disconnect());
  });
});
