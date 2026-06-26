/**
 * Test Suite 02 — File Loading
 * Verifies YAML files can be loaded and the Gantt renders.
 */
describe('02 – File Loading', () => {
  beforeEach(() => {
    cy.visit('/');
    // Open file dialog via menu
    cy.contains('ファイル').click();
    cy.contains('開く').click();
    cy.contains('ファイルを開く').should('be.visible');
  });

  it('file dialog shows EnvConfig and Schedule file inputs', () => {
    cy.contains('EnvConfig').should('be.visible');
    cy.contains('スケジュール').should('be.visible');
  });

  it('loads both YAML fixtures and renders the worker view Gantt', () => {
    cy.get('input[type="file"]').eq(0)
      .selectFile('cypress/fixtures/envConfig.yaml', { force: true });
    cy.get('input[type="file"]').eq(1)
      .selectFile('cypress/fixtures/schedule.yaml', { force: true });
    cy.contains('button', 'OK').click();

    // Gantt grid should appear
    cy.contains('Alice Tanaka', { timeout: 6000 }).should('be.visible');
    cy.contains('Bob Yamada').should('be.visible');
  });

  it('shows status bar file loaded message after load', () => {
    cy.get('input[type="file"]').eq(0)
      .selectFile('cypress/fixtures/envConfig.yaml', { force: true });
    cy.get('input[type="file"]').eq(1)
      .selectFile('cypress/fixtures/schedule.yaml', { force: true });
    cy.contains('button', 'OK').click();

    cy.contains('ファイルを読み込みました', { timeout: 5000 }).should('be.visible');
  });

  it('enables action buttons after load', () => {
    cy.get('input[type="file"]').eq(0)
      .selectFile('cypress/fixtures/envConfig.yaml', { force: true });
    cy.get('input[type="file"]').eq(1)
      .selectFile('cypress/fixtures/schedule.yaml', { force: true });
    cy.contains('button', 'OK').click();

    cy.contains('+ 割付追加', { timeout: 5000 }).should('not.be.disabled');
    cy.contains('+ 新規製番追加').should('not.be.disabled');
  });
});
