# Phase 1 — Personal Azure account + cost guardrails

**Goal of this phase:** an Azure account ready to use, with strict cost
guardrails so you can never accidentally spend more than your budget.

**Time:** ~20 minutes.
**Cost:** $0 (the budget itself is free; resources we'll create in later
phases are mostly free-tier).

---

## What we're setting up and why

Three concepts to know before clicking anything:

| Concept              | What it is                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Subscription**     | Your billing account. Everything you create has to live "in" one subscription. Cost is billed at this level. |
| **Resource Group**   | A folder for related resources. Delete the group → every resource inside dies + billing stops. Critical for cleanup. |
| **Budget**           | An email alert when spending hits a threshold you set (e.g. "email me at $5"). Free to create; doesn't stop charges, just warns. |

**Why one resource group for everything:** at the end of any session, if you
delete the group, *everything* in it is gone and billing stops. No leaked
storage account quietly charging $0.40/month forever.

---

## Step 1 — Create / sign into your personal Azure account

If you don't have one yet:

1. Go to https://azure.microsoft.com/free
2. Sign up with your **personal** Microsoft account (not your work email).
3. You'll be asked for a credit card — Microsoft uses it to verify identity,
   not to bill. Free-tier resources stay free unless you upgrade.
4. You get:
   - **$200 credit** valid for 30 days (use or lose)
   - **12 months free** on popular services (storage, VMs, etc., with limits)
   - **Always-free** services that never expire (Container Apps has one of these)

If you already have an account, skip to Step 2.

After signup, you're at the **Azure Portal**: https://portal.azure.com.
Bookmark it. This is where you'll spend most of your time.

---

## Step 2 — Confirm which subscription you're in

Top-right corner of the portal → click your account avatar.
You'll see either "Free Trial" or "Azure subscription 1" or "Pay-As-You-Go".

Note the **Subscription ID** (a GUID like `12345abc-...`). Save it somewhere —
you'll need it for CLI commands later.

To copy it:
1. Portal search bar (top) → type `subscriptions` → click the result.
2. Click your subscription name.
3. The **Subscription ID** is right at the top — click the copy icon.

---

## Step 3 — Create one resource group for all experiments

1. Portal search → type `resource groups` → click result.
2. Click **+ Create**.
3. Fill in:
   - **Subscription:** the one you just confirmed
   - **Resource group:** `rg-timefold-dev`
   - **Region:** `Japan East` (low latency from your location; pick whatever's closest to you geographically — `East US` and `West Europe` are also common cheap defaults)
4. Click **Review + create** → **Create**.

That's it. The group is just metadata; it costs nothing.

> **Why this name:** `rg-` is the Azure convention for resource groups,
> `timefold` is the project, `dev` means "personal development environment."
> Later when you set up the company subscription you might have `rg-timefold-prod`
> alongside it.

---

## Step 4 — Set a $5 budget with email alerts ⚠️ (most important step)

This is the safety net. Do not skip it.

1. Portal search → `cost management + billing` → click result.
2. In the left sidebar of that page, click **Cost Management**.
3. In the sub-menu, click **Budgets**.
4. At the top, set the **Scope** dropdown to your subscription.
5. Click **+ Add**.
6. Fill in:
   - **Name:** `timefold-personal-budget`
   - **Reset period:** `Monthly`
   - **Creation date:** today
   - **Expiration date:** 12 months from now
   - **Amount:** `5` (USD or your local currency)
7. Click **Next**.
8. On the alerts page, add three alerts so you get warned early:
   - Type: **Actual**, Threshold: `50` %, Action: email to your address
   - Type: **Actual**, Threshold: `90` %, Action: email to your address
   - Type: **Forecasted**, Threshold: `100` %, Action: email to your address
9. Enter your email address in the **Alert recipients** box.
10. Click **Create**.

You'll now get an email if monthly spending crosses $2.50, $4.50, or is
forecasted to hit $5. **It will NOT cut off services** — Azure doesn't
auto-stop. You have to manually delete resources when alerted.

> **Why $5 not $1:** Some Azure free-tier services round up tiny charges to
> ~$0.50, and the portal sometimes shows pending charges before they're
> credited back. A $1 budget would fire false alerts. $5 is loose enough to
> not nag you while still being a real signal if something's wrong.

---

## Step 5 — Install Azure CLI (optional but very useful)

The portal works for everything, but the CLI (`az`) makes follow-along docs
and cleanup much faster. Install it now and you'll thank yourself later.

**Windows:**
1. Download: https://aka.ms/installazurecliwindows
2. Run the MSI installer.
3. Open a **new** Git Bash window (so it picks up PATH changes).
4. Verify:
   ```bash
   az --version
   ```
   You should see version numbers.

Then sign in:
```bash
az login
```
A browser opens — pick your personal account. The CLI saves your credentials
locally.

Confirm the right subscription is active:
```bash
az account show
```
Should print your subscription name and id.

If you have multiple subscriptions and need to switch:
```bash
az account list --output table
az account set --subscription "Free Trial"     # or paste the subscription id
```

---

## Step 6 — Verify the resource group exists from CLI

Just to confirm everything is wired correctly:

```bash
az group show --name rg-timefold-dev --output table
```

Should print one row showing your group in the region you picked.

---

## Cost-control habits going forward

Internalise these — they're the difference between "$0/month" and "$50/month
without realising it":

1. **Always create things INSIDE `rg-timefold-dev`.** Never let resources land
   in a different group, otherwise cleanup is a manual hunt.
2. **At the end of a long session, delete the group** if you're done for now:
   ```bash
   az group delete --name rg-timefold-dev --yes --no-wait
   ```
   *(Creates a clean slate next time. The group itself takes seconds to recreate.)*
3. **Check your spend before bed** if you ran new compute that day:
   - Portal → Cost Management → Cost analysis → "Daily costs"
   - Anything weird? Investigate before sleeping.
4. **Never create a VM**, App Service Plan, AKS cluster, or anything that
   says "always-on" without understanding the hourly cost first. ACA, Blob,
   Functions, Container Instances all scale to zero — those are safe.
5. **Tag resources with `purpose=timefold-poc`** so cost analysis can group
   them. (Optional but nice.)

---

## What you should have at the end of Phase 1

- [x] Azure account with valid login
- [x] Subscription ID written down
- [x] One resource group `rg-timefold-dev` in your chosen region
- [x] $5 monthly budget with 50% / 90% / 100% email alerts
- [x] Azure CLI installed and signed in (`az account show` works)
- [x] Verified the resource group exists via CLI

Tell me when this is all green and we'll move to Phase 2 — creating the
Storage account + blob container.

---

## Troubleshooting

| Symptom                                                     | Cause                                | Fix                                                                  |
| ----------------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------- |
| Sign-up rejects your card                                   | Some virtual / prepaid cards         | Try a real debit/credit card; Microsoft doesn't charge for free tier |
| Portal in wrong language                                    | Browser locale                       | Top-right gear → Settings → Language                                 |
| "You don't have permission" creating budget                 | You're in a Free Trial subscription with limited features | Most Free Trial accounts can create budgets; if not, upgrade to Pay-As-You-Go (still free if you don't use anything) |
| `az: command not found`                                     | CLI installer didn't update PATH     | Close and reopen Git Bash; check `where az` shows a path             |
| `az login` opens browser but hangs                          | Browser/account mismatch             | Sign out of other Microsoft accounts in your browser first           |
