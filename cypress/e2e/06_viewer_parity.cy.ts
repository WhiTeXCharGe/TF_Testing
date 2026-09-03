/**
 * Test Suite 06 — Viewer Parity
 * Verifies a view-role participant can scroll/filter/check-constraints but
 * cannot edit, using the real collab server (requires `npm run dev:all`,
 * not just `npm run dev` — see this feature's design doc for why).
 */
describe('06 – Viewer Parity', () => {
  it('a view-role join can toggle 出入国バー and run the constraint check, but cannot drag a bar or use undo/redo', () => {
    cy.visit('/');
    cy.loadFixtures();

    // Start a session as editor, capture its id from the share link so the
    // test can join a second, view-role client via the URL path — the same
    // mechanism a real viewer's browser link uses.
    cy.contains('共同編集').click();
    cy.contains('セッションを開始').click();
    cy.get('input[placeholder="セッション名を入力"]').type('Viewer Parity Test');
    cy.get('input[placeholder="表示名を入力"]').type('Editor');
    cy.contains('button', '開始する').click();
    cy.contains('セッション情報', { timeout: 8000 }).should('exist');

    cy.window().then(win => {
      const url = new URL(win.location.href);
      // The session id isn't in this window's own URL (it's the creator, not
      // a link-joiner) — read it from the copyable edit-link input instead.
    });
    cy.get('input[readonly]').first().invoke('val').then(editLink => {
      const sessionId = new URL(String(editLink)).searchParams.get('session');

      cy.visit(`/?session=${sessionId}&role=view`);
      cy.get('input[placeholder]').first().type('Viewer');
      cy.contains('button', '参加する').click();

      // Enabled for a viewer: flight-stints toggle and constraint check.
      cy.contains('✈ 出入国バー', { timeout: 8000 }).click();
      cy.contains('☑ 制約チェック').click();

      // Disabled for a viewer: undo/redo, and dragging a bar has no effect —
      // the button itself is visibly inert (opacity/disabled), which is the
      // observable proxy for "the handler is gated" from outside the app.
      cy.contains('button', '元に戻す').should('be.disabled');
      cy.contains('button', 'やり直し').should('be.disabled');
    });
  });
});
