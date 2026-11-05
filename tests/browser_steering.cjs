// NODE_PATH=<playwright node_modules> node tests/browser_steering.cjs
const { chromium } = require('@playwright/test');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
  const page = await browser.newPage();
  page.setDefaultTimeout(15000);
  try {
    await page.goto('http://127.0.0.1:19621');
    const input = page.getByRole('textbox', {name:'Type your message here...'});
    await input.waitFor();
    await input.fill('Research this fixture');
    await input.press('Enter');
    const steering = page.getByRole('textbox', {name:'Guide this response…'});
    await steering.waitFor();
    await page.waitForFunction(() => !document.querySelector('textarea[placeholder="Guide this response…"]')?.disabled);
    await steering.fill('Focus only on FAIR');
    await steering.press('Enter');
    await page.getByText('Received: Focus only on FAIR', {exact:true}).waitFor();
    await page.getByText('Fixture answer: Focus only on FAIR', {exact:true}).waitFor({timeout:15000});
    await input.waitFor();
    assert.equal(await page.getByText('Focus only on FAIR', {exact:true}).count(), 1);
    assert.equal(await page.getByText('Fixture answer: Focus only on FAIR', {exact:true}).count(), 1);
    await input.fill('Research a slow fixture');
    await input.press('Enter');
    const stop = page.getByRole('button', {name:'Stop', exact:true});
    await stop.waitFor();
    const started = Date.now();
    await stop.click();
    await page.getByText('Response stopped.', {exact:true}).waitFor({timeout:10000});
    assert.ok(Date.now()-started < 10000, 'Stop waited for the slow tool');
    await input.waitFor();
    await input.fill('Another fixture');
    await input.press('Enter');
    await page.getByText('Fixture answer: original scope', {exact:true}).waitFor({timeout:15000});
    await input.waitFor();
    await input.fill('Please ask about fixture scope');
    await input.press('Enter');
    await steering.waitFor();
    await steering.fill('Use only the NeXus evidence');
    await steering.press('Enter');
    await page.getByText('Received: Use only the NeXus evidence', {exact:true}).waitFor();
    await page.getByText('Which fixture scope?', {exact:true}).waitFor();
    await page.getByRole('button', {name:'FAIR', exact:true}).click();
    await page.getByText('Fixture answer: Use only the NeXus evidence', {exact:true}).waitFor();
    await input.waitFor();
    await input.fill('Research then request approval');
    await input.press('Enter');
    await steering.waitFor();
    await steering.fill('Keep the command brief');
    await steering.press('Enter');
    await page.getByText('Received: Keep the command brief', {exact:true}).waitFor();
    await page.getByText('Allow this command?', {exact:true}).waitFor();
    await page.getByRole('button', {name:/\bRun$/}).click();
    await page.getByText('Fixture answer: Keep the command brief', {exact:true}).waitFor();
    assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
    console.log('Browser acceptance passed: steering, one answer, stalled-tool Stop, fresh next turn, steering plus clarification and approval resumes.');
  } catch (error) {
    await page.screenshot({path:'/tmp/juena-steering-browser-failure.png', fullPage:true});
    console.error((await page.locator('body').innerText()).slice(-6000));
    throw error;
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode=1; });
