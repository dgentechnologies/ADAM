async (page) => {
  await page.route('https://identitytoolkit.googleapis.com/**', route => route.fulfill({
    status: 400, contentType: 'application/json', body: JSON.stringify({ error: { code: 400, message: 'INVALID_LOGIN_CREDENTIALS' } }),
  }));
  await page.getByRole('textbox', { name: 'Email', exact: true }).fill('android-qa@example.invalid');
  await page.getByRole('textbox', { name: 'Password', exact: true }).fill('Invalid-test-password');
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await page.getByText('The email or password is incorrect.', { exact: true }).first().waitFor();
  if (new URL(page.url()).pathname !== '/sign-in/') throw new Error('Failed login incorrectly advanced setup');
  const setup = await page.evaluate(() => JSON.parse(localStorage.getItem('adam.setup.v1')));
  if (setup.state.signedIn) throw new Error('Failed login persisted a fake authenticated state');
  await page.screenshot({ path: 'output/playwright/android22-email-error.png' });
  await page.unroute('https://identitytoolkit.googleapis.com/**');
  return { failedLoginStayedOnSignIn: true, savedSignedIn: setup.state.signedIn };
}
