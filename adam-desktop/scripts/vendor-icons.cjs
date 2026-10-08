// Generate the tiny local sprite from the project's pinned Lucide installation.
const fs = require('fs');
const path = require('path');
const { createRequire } = require('module');
const root = path.resolve(__dirname, '..');
const fromWeb = createRequire(path.resolve(root, '../adam-mobile/apps/web/package.json'));
const React = fromWeb('react');
const { renderToStaticMarkup } = fromWeb('react-dom/server');
const icons = fromWeb('lucide-react');
const names = {home:'House',sliders:'SlidersHorizontal',devices:'MonitorSmartphone',laptop:'Laptop',clock:'Clock3',memory:'Brain',code:'SquareTerminal',activity:'Activity',settings:'Settings2',user:'UserRound',pause:'Pause',play:'Play',arrow:'ArrowRight',refresh:'RefreshCw',rotate:'RotateCcw',search:'Search',trash:'Trash2',plus:'Plus',close:'X',key:'KeyRound',cpu:'Cpu',sync:'RefreshCw',sparkles:'Sparkles',edit:'Pencil',check:'Check',chevron:'ChevronDown',shield:'ShieldCheck',volume:'Volume2',mute:'VolumeX',sun:'Sun',clipboard:'Clipboard',lock:'LockKeyhole',next:'SkipForward',previous:'SkipBack'};
const symbols = Object.entries(names).map(([id,name])=>{
  const svg=renderToStaticMarkup(React.createElement(icons[name],{size:24,strokeWidth:1.7}));
  return `<symbol id="${id}" viewBox="0 0 24 24">${svg.replace(/^<svg[^>]*>/,'').replace(/<\/svg>$/,'')}</symbol>`;
});
const mobileSignIn = fs.readFileSync(path.resolve(root,'../adam-mobile/apps/web/src/app/(setup)/sign-in/page.tsx'),'utf8');
const paths=[...mobileSignIn.matchAll(/<path\s+d="([^"]+)"\s+fill="(#[A-Fa-f0-9]+)"\s*\/>/g)];
if(paths.length!==4)throw new Error('Expected the four Google brand paths');
symbols.push(`<symbol id="google" viewBox="0 0 24 24">${paths.map(([,d,color])=>`<path d="${d}" fill="${color}" stroke="none"/>`).join('')}</symbol>`);
fs.writeFileSync(path.join(root,'resources/static/images/icons.svg'),`<svg xmlns="http://www.w3.org/2000/svg">${symbols.join('\n')}</svg>\n`);
const lucideRoot=path.resolve(fromWeb.resolve('lucide-react'),'../../../');
fs.mkdirSync(path.join(root,'resources/static/licenses'),{recursive:true});
fs.copyFileSync(path.join(lucideRoot,'LICENSE'),path.join(root,'resources/static/licenses/LUCIDE.txt'));
console.log(`Vendored ${symbols.length} icons. Lucide 0.469.0; Google mark retained from mobile.`);
