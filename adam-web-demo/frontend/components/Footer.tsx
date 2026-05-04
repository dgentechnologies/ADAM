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
    <footer className="border-t border-[rgba(29,29,31,0.14)] bg-[rgba(223,225,231,0.76)] mt-16">
      <div className="max-w-7xl mx-auto px-6 py-14 grid grid-cols-2 md:grid-cols-4 gap-10">
        <div className="col-span-2 md:col-span-1 space-y-4">
          <Link href="/"><Image src="/images/logo.png" alt="DGEN Technologies" width={120} height={40} /></Link>
          <p className="text-[rgba(29,29,31,0.6)] text-sm leading-relaxed">
            Innovate. Integrate. Inspire.<br />Made in India · Kolkata, 2025
          </p>
        </div>

        <div>
          <h4 className="text-xs font-semibold uppercase tracking-widest text-[rgba(29,29,31,0.56)] mb-4">Company</h4>
          <ul className="space-y-2">
            {COMPANY_LINKS.map(({ label, href }) => (
              <li key={href}><Link href={href} className="text-[rgba(29,29,31,0.66)] hover:text-[rgba(29,29,31,0.95)] text-sm transition">{label}</Link></li>
            ))}
          </ul>
        </div>

        <div>
          <h4 className="text-xs font-semibold uppercase tracking-widest text-[rgba(29,29,31,0.56)] mb-4">Products</h4>
          <ul className="space-y-2">
            {PRODUCT_LINKS.map(({ label, href }) => (
              <li key={href}>
                <Link href={href} className={`text-sm transition ${label === 'ADAM' ? 'text-[#0a84ff] hover:text-[#0066cc]' : 'text-[rgba(29,29,31,0.66)] hover:text-[rgba(29,29,31,0.95)]'}`}>
                  {label}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        <div className="space-y-6">
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-[rgba(29,29,31,0.56)] mb-4">Legal</h4>
            <ul className="space-y-2">
              {LEGAL_LINKS.map(({ label, href }) => (
                <li key={href}><Link href={href} className="text-[rgba(29,29,31,0.66)] hover:text-[rgba(29,29,31,0.95)] text-sm transition">{label}</Link></li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="text-xs font-semibold uppercase tracking-widest text-[rgba(29,29,31,0.56)] mb-4">Connect</h4>
            <ul className="space-y-2">
              {SOCIAL_LINKS.map(({ label, href }) => (
                <li key={href}>
                  <a href={href} target="_blank" rel="noopener noreferrer" className="text-[rgba(29,29,31,0.66)] hover:text-[rgba(29,29,31,0.95)] text-sm transition">{label}</a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="border-t border-[rgba(29,29,31,0.1)] px-6 py-5 text-center text-xs text-[rgba(29,29,31,0.5)]">
        © {new Date().getFullYear()} DGEN Technologies Pvt. Ltd. All rights reserved.
      </div>
    </footer>
  );
}
