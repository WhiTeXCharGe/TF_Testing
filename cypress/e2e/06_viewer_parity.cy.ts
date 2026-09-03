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

    // The session id isn't in this window's own URL (it's the creator, not
    // a link-joiner) — read it from the copyable edit-link input instead.
    cy.get('input[readonly]').first().invoke('val').then(editLink => {
      const sessionId = new URL(String(editLink)).searchParams.get('session');

      cy.visit(`/?session=${sessionId}&role=view`);
      cy.get('input[placeholder]').first().type('Viewer');
      cy.contains('button', '参加する').click();

      // Enabled for a viewer: flight-stints toggle and constraint check.
      cy.contains('✈ 出入国バー', { timeout: 8000 }).click();
      cy.contains('☑ 制約チェック').click();

      // Disabled for a viewer: undo/redo.
      cy.contains('button', '元に戻す').should('be.disabled');
      cy.contains('button', 'やり直し').should('be.disabled');

      // Disabled for a viewer: actually attempt to drag an assignment bar
      // (mousedown on the bar, mousemove, mouseup — the same event sequence
      // `startDrag`'s window listeners react to) and assert the bar's
      // position/worker/date-range attributes are unchanged afterwards.
      // This is the direct proxy for the `readOnly` guard at the top of
      // `startDrag` actually firing, not just that undo/redo are disabled.
      cy.get('[data-testid="assignment-bar"]', { timeout: 8000 }).should('have.length.greaterThan', 0);

      cy.get('[data-testid="assignment-bar"]').first().then($bar => {
        const before = {
          startIndex: $bar.attr('data-start-index'),
          endIndex: $bar.attr('data-end-index'),
          workerId: $bar.attr('data-worker-id'),
          left: $bar[0].style.left,
        };
        const rect = $bar[0].getBoundingClientRect();
        const startX = rect.left + rect.width / 2;
        const startY = rect.top + rect.height / 2;

        // A small gap between each triggered event gives React's event
        // handling (which registers the window mousemove/mouseup listeners
        // inside the mousedown handler) time to actually take effect before
        // the next synthetic event fires — firing all three back-to-back in
        // the same tick is unreliable even against the real (non-gated)
        // drag handler.
        cy.wrap($bar).trigger('mousedown', { clientX: startX, clientY: startY, button: 0, force: true });
        cy.wait(50);
        cy.get('body').trigger('mousemove', { clientX: startX + 200, clientY: startY, force: true });
        cy.wait(50);
        cy.get('body').trigger('mouseup', { force: true });
        cy.wait(300);

        // No drag-preview overlay should ever have appeared — the guard
        // returns before any preview state is set.
        cy.get('body').find('[style*="dashed"]').should('not.exist');

        cy.get('[data-testid="assignment-bar"]').first().then($after => {
          expect($after.attr('data-start-index')).to.equal(before.startIndex);
          expect($after.attr('data-end-index')).to.equal(before.endIndex);
          expect($after.attr('data-worker-id')).to.equal(before.workerId);
          expect($after[0].style.left).to.equal(before.left);
        });
      });
    });
  });
});
