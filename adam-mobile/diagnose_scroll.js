const { spawn } = require('child_process');

async function testPage(urlPath) {
  const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';
  const chrome = spawn(chromePath, [
    '--headless=new',
    '--remote-debugging-port=9222',
    '--disable-gpu',
    '--window-size=390,844',
    '--user-data-dir=C:\\temp\\chrome-test-profile'
  ]);

  // Wait for Chrome to start
  await new Promise(r => setTimeout(r, 1500));

  try {
    const listRes = await fetch('http://127.0.0.1:9222/json');
    const tabs = await listRes.json();
    const wsUrl = tabs[0].webSocketDebuggerUrl;

    const ws = new WebSocket(wsUrl);
    await new Promise((res, rej) => {
      ws.onopen = res;
      ws.onerror = rej;
    });

    let msgId = 1;
    function send(method, params = {}) {
      return new Promise((resolve) => {
        const id = msgId++;
        const handler = (evt) => {
          const data = JSON.parse(evt.data);
          if (data.id === id) {
            ws.removeEventListener('message', handler);
            resolve(data.result);
          }
        };
        ws.addEventListener('message', handler);
        ws.send(JSON.stringify({ id, method, params }));
      });
    }

    await send('Page.enable');
    await send('Page.navigate', { url: `http://localhost:3000${urlPath}` });
    await new Promise(r => setTimeout(r, 2000));

    const evalResult = await send('Runtime.evaluate', {
      expression: `(() => {
        const html = document.documentElement;
        const body = document.body;
        const main = document.querySelector('main');
        const header = document.querySelector('header');
        const nav = document.querySelector('nav');

        function getInfo(el, name) {
          if (!el) return null;
          const style = window.getComputedStyle(el);
          return {
            name,
            tagName: el.tagName,
            className: el.className,
            clientHeight: el.clientHeight,
            scrollHeight: el.scrollHeight,
            offsetHeight: el.offsetHeight,
            overflowX: style.overflowX,
            overflowY: style.overflowY,
            position: style.position,
            height: style.height,
            minHeight: style.minHeight,
            maxHeight: style.maxHeight,
            touchAction: style.touchAction,
            pointerEvents: style.pointerEvents,
          };
        }

        // Test scrolling
        const initialScrollY = window.scrollY;
        const initialBodyScrollTop = body.scrollTop;
        const initialHtmlScrollTop = html.scrollTop;
        const initialMainScrollTop = main ? main.scrollTop : 0;

        window.scrollTo(0, 500);
        body.scrollTop = 500;
        html.scrollTop = 500;
        if (main) main.scrollTop = 500;

        const afterWindowScrollY = window.scrollY;
        const afterBodyScrollTop = body.scrollTop;
        const afterHtmlScrollTop = html.scrollTop;
        const afterMainScrollTop = main ? main.scrollTop : 0;

        return {
          windowInnerHeight: window.innerHeight,
          windowInnerWidth: window.innerWidth,
          scrollingElement: document.scrollingElement.tagName,
          html: getInfo(html, 'html'),
          body: getInfo(body, 'body'),
          main: getInfo(main, 'main'),
          header: getInfo(header, 'header'),
          nav: getInfo(nav, 'nav'),
          scrollTest: {
            initial: { window: initialScrollY, body: initialBodyScrollTop, html: initialHtmlScrollTop, main: initialMainScrollTop },
            after: { window: afterWindowScrollY, body: afterBodyScrollTop, html: afterHtmlScrollTop, main: afterMainScrollTop }
          }
        };
      })()`,
      returnByValue: true
    });

    console.log(`\n=== RESULTS FOR ${urlPath} ===`);
    console.log(JSON.stringify(evalResult.result.value, null, 2));

    ws.close();
  } catch (err) {
    console.error('Error during test:', err);
  } finally {
    chrome.kill();
  }
}

async function run() {
  await testPage('/settings/');
  await testPage('/gallery/');
  process.exit(0);
}

run();
