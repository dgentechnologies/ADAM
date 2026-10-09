async (page) => {
  const original = await page.evaluate(() => localStorage.getItem('adam.setup.v1'));
  await page.evaluate(() => localStorage.setItem('adam.setup.v1', '{broken-setup'));
  await page.goto('http://localhost:3000/');
  await page.waitForURL('**/welcome/', { timeout: 15000 });
  const recovered = await page.evaluate(() => Object.keys(localStorage).some(key => key.startsWith('adam.setup.v1.recovery.') && localStorage.getItem(key) === '{broken-setup'));
  if (!recovered) throw new Error('Damaged setup was not backed up');
  const saved = JSON.parse(original);
  saved.state.currentStep = 'name-device';
  saved.state.completedAt = null;
  await page.evaluate(value => localStorage.setItem('adam.setup.v1', value), JSON.stringify(saved));
  await page.goto('http://localhost:3000/');
  await page.waitForURL('**/name-device/', { timeout: 15000 });
  await page.getByRole('heading', { name: 'What should we call him?' }).waitFor();
  await page.screenshot({ path: 'output/playwright/android22-resumed-naming.png' });
  const blocked = await page.context().newPage();
  await blocked.addInitScript(() => {
    const get = Storage.prototype.getItem;
    Storage.prototype.getItem = function(key) { if (key === 'adam.setup.v1') throw new Error('QA storage unavailable'); return get.call(this, key); };
  });
  await blocked.goto('http://localhost:3000/');
  await blocked.getByRole('button', { name: 'Try again', exact: true }).waitFor();
  await blocked.screenshot({ path: 'output/playwright/android22-storage-retry.png' });
  await blocked.close();
  await page.evaluate(value => localStorage.setItem('adam.setup.v1', value), original);
  return { corruptSetupRecoveredWithoutLoss: recovered, savedNamingStepResumed: true, storageFailureShowsRetry: true };
}
