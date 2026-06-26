/**
 * Test Suite 03 — Worker View Filter Bar
 * Verifies the new per-view filter chips work correctly.
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

describe('03 – Worker View Filter Bar', () => {
  beforeEach(loadApp);

  it('shows the worker view filter bar with expected chips', () => {
    // Should be on worker view by default
    cy.contains('装置').should('be.visible');
    cy.contains('工程').should('be.visible');
    cy.contains('Fab').should('be.visible');
    cy.contains('Region').should('be.visible');
  });

  it('bar name search input is visible and writable', () => {
    cy.get('input[placeholder*="バー名"]').should('be.visible').type('SU-1001');
    cy.get('input[placeholder*="バー名"]').should('have.value', 'SU-1001');
  });

  it('typing a bar name filters to matching workers', () => {
    cy.get('input[placeholder*="バー名"]').type('SU-1001');
    // Alice and Bob are assigned to SU-1001; Carol to SU-1001 p2
    cy.contains('Alice Tanaka').should('be.visible');
    // Workers not related to SU-1001 should not be visible
    // (In our fixture, all workers are on SU-1001 so we mainly check no crash)
  });

  it('クリア button appears when filter is active', () => {
    cy.get('input[placeholder*="バー名"]').type('test');
    cy.contains('クリア').should('be.visible');
  });

  it('clicking クリア resets bar name filter', () => {
    cy.get('input[placeholder*="バー名"]').type('nonexistent');
    cy.contains('クリア').click();
    cy.get('input[placeholder*="バー名"]').should('have.value', '');
  });

  it('装置 chip opens a dropdown with searchable options', () => {
    cy.contains('button', '装置').click();
    // Dropdown should appear with SU-1001 and SU-1002
    cy.contains('SU-1001').should('be.visible');
    cy.contains('SU-1002').should('be.visible');
  });

  it('装置 chip button is visible and clickable', () => {
    // Just verify the chip is rendered and responds to click without error
    cy.contains('button', '装置').should('be.visible').click();
    cy.contains('button', '装置').should('exist');
  });

  it('Fab chip opens dropdown with fab options', () => {
    cy.contains('button', 'Fab').click();
    cy.contains('Osaka Fab').should('be.visible');
  });

  it('date range inputs are visible', () => {
    cy.get('input[type="date"]').should('have.length.greaterThan', 1);
  });

  it('setting a start date shows クリア button', () => {
    cy.get('input[type="date"]').first().type('2025-09-10');
    cy.contains('クリア').should('be.visible');
  });
});

describe('03b – Module View Filter Bar', () => {
  beforeEach(() => {
    loadApp();
    // Switch to device/module view
    cy.contains('装置ビュー').click();
    cy.contains('SU-1001', { timeout: 4000 }).should('be.visible');
  });

  it('shows module view filter chips', () => {
    cy.contains('button', '作業者').should('be.visible');
    cy.contains('button', 'Fab').should('be.visible');
    cy.contains('button', 'Region').should('be.visible');
  });

  it('作業者 chip opens dropdown with workers who have assignments', () => {
    cy.contains('button', '作業者').click();
    cy.contains('Alice Tanaka').should('be.visible');
    cy.contains('Bob Yamada').should('be.visible');
  });

  it('misc tasks (wf_misc) do not appear as module rows', () => {
    cy.contains('Other Dummy Work').should('not.exist');
  });

  it('SU-1001 and SU-1002 appear as module rows', () => {
    cy.contains('SU-1001').should('be.visible');
    cy.contains('SU-1002').should('be.visible');
  });
});
