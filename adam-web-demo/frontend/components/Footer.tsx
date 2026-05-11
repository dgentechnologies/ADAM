import Link from 'next/link';
import Image from 'next/image';

const COMPANY_LINKS = [
  { label: 'About Us', href: '/about' },
  { label: 'Careers',  href: '/careers' },
  { label: 'Blog',     href: '/blog' },
  { label: 'Contact',  href: '/contact' },
];

const PRODUCT_LINKS = [
  { label: 'ADAM',              href: '/adam' },
  { label: 'Auralis Ecosystem', href: '/products/auralis-ecosystem' },
  { label: 'Solar Street Light',href: '/products/solar-street-light' },
  { label: 'LED Street Light',  href: '/products/led-street-light' },
];

const LEGAL_LINKS = [
  { label: 'Privacy Policy',   href: '/privacy-policy' },
  { label: 'Terms of Service', href: '/terms-of-service' },
  { label: 'FAQ',              href: '/faq' },
];

const SOCIAL_LINKS = [
  { label: 'LinkedIn',   href: 'https://linkedin.com/company/dgentechnologies' },
  { label: 'X / Twitter',href: 'https://x.com/dgen_tec' },
  { label: 'Instagram',  href: 'https://instagram.com/dgen_technologies' },
  { label: 'YouTube',    href: 'https://youtube.com/@DGENTECHNOLOGIES' },
];

export function Footer() {
  return (
    <footer className="mt-16 border-t border-white/10 bg-[rgba(8,10,12,0.9)]">
      <div className="mx-auto grid max-w-7xl grid-cols-2 gap-10 px-6 py-14 md:grid-cols-4">
        <div className="col-span-2 md:col-span-1 space-y-4">
          <Link href="/"><Image src="/images/logo.png" alt="DGEN Technologies" width={120} height={40} /></Link>
          <p className="text-sm leading-relaxed text-white/58">
            Innovate. Integrate. Inspire.<br />Made in India · Kolkata, 2025
          </p>
        </div>

        <div>
          <h4 className="mb-4 text-xs font-semibold uppercase tracking-widest text-white/40">Company</h4>
          <ul className="space-y-2">
            {COMPANY_LINKS.map(({ label, href }) => (
              <li key={href}><Link href={href} className="text-sm text-white/62 transition hover:text-white">{label}</Link></li>
            ))}
          </ul>
        </div>

        <div>
          <h4 className="mb-4 text-xs font-semibold uppercase tracking-widest text-white/40">Products</h4>
          <ul className="space-y-2">
            {PRODUCT_LINKS.map(({ label, href }) => (
              <li key={href}>
                <Link href={href} className={`text-sm transition ${label === 'ADAM' ? 'text-[#4af0ff] hover:text-[#8af6ff]' : 'text-white/62 hover:text-white'}`}>
                  {label}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        <div className="space-y-6">
          <div>
            <h4 className="mb-4 text-xs font-semibold uppercase tracking-widest text-white/40">Legal</h4>
            <ul className="space-y-2">
              {LEGAL_LINKS.map(({ label, href }) => (
                <li key={href}><Link href={href} className="text-sm text-white/62 transition hover:text-white">{label}</Link></li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="mb-4 text-xs font-semibold uppercase tracking-widest text-white/40">Connect</h4>
            <ul className="space-y-2">
              {SOCIAL_LINKS.map(({ label, href }) => (
                <li key={href}>
                  <a href={href} target="_blank" rel="noopener noreferrer" className="text-sm text-white/62 transition hover:text-white">{label}</a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="border-t border-white/10 px-6 py-5 text-center text-xs text-white/42">
        © {new Date().getFullYear()} DGEN Technologies Pvt. Ltd. All rights reserved.
      </div>
    </footer>
  );
}
