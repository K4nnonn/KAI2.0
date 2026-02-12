const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');

(async () => {
  const htmlPath = path.resolve(__dirname, 'owner_readiness_checklist.html');
  const pdfPath = path.resolve(__dirname, 'owner_readiness_checklist.pdf');
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: 'networkidle' });
  await page.pdf({ path: pdfPath, format: 'A4', printBackground: true, margin: { top: '20px', bottom: '20px', left: '20px', right: '20px' } });
  await browser.close();
  console.log('PDF written:', pdfPath);
})();
