/**
 * Test Suite 01 — Empty State
 * Verifies the app loads correctly before any file is loaded.
 */
describe('01 – Empty State', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  it('renders the app title in the menu bar', () => {
    cy.contains('Gantt Chart Editor').should('be.visible');
  });

  it('shows the "ファイル" menu', () => {
    cy.contains('ファイル').should('be.visible');
  });

  it('shows the toolbar with action buttons', () => {
    cy.contains('+ 割付追加').should('be.visible');
    cy.contains('+ 新規製番追加').should('be.visible');
  });

  it('shows the view toggle buttons', () => {
    cy.contains('ワーカービュー').should('be.visible');
    cy.contains('装置ビュー').should('be.visible');
  });

  it('shows the empty-state message prompting to load a file', () => {
    cy.contains('ファイル').should('be.visible');
    cy.contains('開く').should('exist');
  });

  it('action buttons are disabled when no file is loaded', () => {
    cy.contains('+ 割付追加').should('be.disabled');
    cy.contains('+ 新規製番追加').should('be.disabled');
  });

  it('status bar shows "ファイル未読み込み"', () => {
    cy.contains('ファイル未読み込み').should('be.visible');
  });

  it('clicking "ファイル" → "開く" opens the file dialog', () => {
    cy.contains('ファイル').click();
    cy.contains('開く').click();
    cy.contains('ファイルを開く').should('be.visible');
  });

  it('file dialog can be cancelled', () => {
    cy.contains('ファイル').click();
    cy.contains('開く').click();
    cy.contains('キャンセル').click();
    cy.contains('ファイルを開く').should('not.exist');
  });
});
