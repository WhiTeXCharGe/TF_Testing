// Custom Cypress commands

/**
 * Load both fixture YAMLs into the app via the hidden file inputs.
 * Call this after cy.visit('/') to put the app into a data-loaded state.
 */
Cypress.Commands.add('loadFixtures', () => {
  // Open file dialog via the ファイル menu — '開く' is a dropdown item inside
  // that menu, not a bare button on the page.
  cy.contains('ファイル').click();
  cy.contains('開く').click();

  // Wait for dialog
  cy.contains('ファイルを開く').should('be.visible');

  // Load EnvConfig fixture into its input
  cy.get('input[type="file"]').eq(0).selectFile('cypress/fixtures/envConfig.yaml', { force: true });
  // Load Schedule fixture into its input
  cy.get('input[type="file"]').eq(1).selectFile('cypress/fixtures/schedule.yaml', { force: true });

  // Confirm load
  cy.contains('button', 'OK').click();

  // Wait for Gantt to appear
  cy.contains('SU-1001', { timeout: 5000 }).should('exist');
});

declare global {
  namespace Cypress {
    interface Chainable {
      loadFixtures(): Chainable<void>;
    }
  }
}
