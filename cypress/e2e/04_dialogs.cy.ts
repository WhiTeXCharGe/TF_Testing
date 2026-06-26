/**
 * Test Suite 04 — Dialogs
 * Verifies that dialogs open, render correctly, and can be dismissed.
 */

function loadApp() {
  cy.visit('/');
  cy.contains('ファイル').click();
  cy.contains('開く').click();
  cy.get('input[type="file"]').eq(0).selectFile('cypress/fixtures/envConfig.yaml', { force: true });
  cy.get('input[type="file"]').eq(1).selectFile('cypress/fixtures/schedule.yaml', { force: true });
  cy.contains('button', 'OK').click();
  cy.contains('Alice Tanaka', { timeout: 6000 }).should('be.visible');
}

describe('04 – 割付追加 Dialog', () => {
  beforeEach(loadApp);

  it('opens the 割付追加 dialog', () => {
    cy.contains('+ 割付追加').click();
    cy.contains('割付追加').should('be.visible');
  });

  it('dialog shows 追加タイプ選択 options', () => {
    cy.contains('+ 割付追加').click();
    cy.contains('通常ワークフロー').should('be.visible');
    cy.contains('休日・不在設定').should('be.visible');
  });

  it('dialog can be cancelled', () => {
    cy.contains('+ 割付追加').click();
    cy.get('button').contains('キャンセル').click();
    cy.contains('通常ワークフロー').should('not.exist');
  });

  it('shows 装置 label in dialog', () => {
    cy.contains('+ 割付追加').click();
    cy.contains('装置').should('be.visible');
  });
});

describe('04 – 新規製番追加 Dialog', () => {
  beforeEach(loadApp);

  it('opens the 新規製番追加 dialog', () => {
    cy.contains('+ 新規製番追加').click();
    cy.contains('新規製番追加').should('be.visible');
  });

  it('shows form tab and upload tab', () => {
    cy.contains('+ 新規製番追加').click();
    cy.contains('フォームで追加').should('be.visible');
    cy.contains('ファイルからインポート').should('be.visible');
  });

  it('workflow dropdown does not include wf_misc (Other Work)', () => {
    cy.contains('+ 新規製番追加').click();
    // The workflow select shows valid workflows — wf_misc should not appear
    cy.contains('ワークフロー').should('be.visible');
    cy.contains('Other Work').should('not.exist');
  });

  it('dialog closes after キャンセル', () => {
    cy.contains('+ 新規製番追加').click();
    cy.contains('フォームで追加').should('be.visible');
    cy.get('button').contains('キャンセル').click();
    cy.contains('フォームで追加').should('not.exist');
  });
});

describe('04 – Undo / Redo', () => {
  beforeEach(loadApp);

  it('Undo button is visible in toolbar', () => {
    cy.contains('元に戻す').should('exist');
  });

  it('Redo button is visible in toolbar', () => {
    cy.contains('やり直し').should('exist');
  });
});
