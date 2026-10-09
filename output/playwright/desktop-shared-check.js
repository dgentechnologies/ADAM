async (page) => {
  await page.getByRole('button', { name: 'Planner', exact: true }).click();
  await page.getByRole('textbox', { name: 'New to-do', exact: true }).first().fill('Shared QA task');
  await page.getByRole('button', { name: 'Add to-do', exact: true }).first().click();
  await page.getByText('Shared QA task', { exact: true }).waitFor();
  await page.getByRole('combobox', { name: 'Type', exact: true }).selectOption('timer');
  await page.getByRole('textbox', { name: 'Label', exact: true }).fill('Shared QA timer');
  await page.getByRole('button', { name: 'Add plan', exact: true }).click();
  await page.getByText('Shared QA timer', { exact: true }).waitFor();
  await page.screenshot({ path: 'output/playwright/desktop22-planner.png' });
  await page.getByRole('button', { name: 'Connection', exact: true }).click();
  await page.getByRole('textbox', { name: 'Device name', exact: true }).fill('Studio QA');
  await page.getByRole('button', { name: 'Add simulated ADAM', exact: true }).click();
  await page.getByText('Studio QA', { exact: true }).waitFor();
  const desk = page.getByRole('article').filter({ has: page.getByText('Desk QA', { exact: true }) }).last();
  await desk.getByRole('button', { name: 'Edit', exact: true }).click();
  await page.getByRole('textbox', { name: 'Device name', exact: true }).fill('Library QA');
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  await page.getByText('Library QA', { exact: true }).waitFor();
  await page.getByRole('button', { name: 'Sync simulated BLE', exact: true }).first().click();
  await page.getByText('Simulated BLE transfer complete.', { exact: true }).waitFor();
  await page.reload();
  await page.getByText('Library QA', { exact: true }).waitFor();
  await page.getByText('Studio QA', { exact: true }).waitFor();
  await page.screenshot({ path: 'output/playwright/desktop22-devices.png' });
  const result = await page.evaluate(async () => {
    const headers = { 'X-ADAM-Session': document.querySelector('meta[name="adam-session"]').content };
    const data = {};
    for (const kind of ['todos', 'clocks', 'devices']) data[kind] = (await (await fetch('/companion/' + kind, { headers })).json()).items;
    return { todos: data.todos.length, clocks: data.clocks.length, devices: data.devices.map(d => d.name), fitsWidth: document.documentElement.scrollWidth <= innerWidth + 1 };
  });
  if (result.todos < 1 || result.clocks < 1 || result.devices.length < 2 || !result.fitsWidth) throw new Error(JSON.stringify(result));
  return result;
}
