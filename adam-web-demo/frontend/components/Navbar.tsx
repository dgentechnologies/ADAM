'use client';

import Link from 'next/link';
import Image from 'next/image';
import { useAuth } from '@/components/FirebaseAuthProvider';

const NAV_LINKS = [
  { label: 'Home',     href: '/' },
  { label: 'About Us', href: '/about' },
  { label: 'Services', href: '/services' },
  { label: 'Products', href: '/products' },
  { label: 'ADAM',     href: '/adam' },
  { label: 'Blog',     href: '/blog' },
  { label: 'Careers',  href: '/careers' },
  { label: 'Contact',  href: '/contact' },
];

export function Navbar() {
  const { user, loading, signOut } = useAuth();

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[rgba(8,10,12,0.78)] backdrop-blur-xl">
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-6 px-6">
        {/* Logo */}
        <Link href="/" className="shrink-0">
          <Image src="/images/logo.png" alt="DGEN Technologies" width={120} height={40} priority />
        </Link>

        {/* Desktop links */}
        <ul className="hidden items-center gap-6 text-sm font-medium text-white/58 lg:flex">
          {NAV_LINKS.map(({ label, href }) => (
            <li key={href}>
              <Link
                href={href}
                className={`transition hover:text-white ${
                  label === 'ADAM' ? 'text-[#4af0ff] hover:text-[#8af6ff]' : ''
                }`}
              >
                {label}
              </Link>
            </li>
          ))}
        </ul>

        {/* Right side */}
        <div className="flex items-center gap-3 shrink-0">
          <Link
            href="/adam/demo"
            className="hidden rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-sm font-semibold text-cyan-100 shadow-[0_0_0_1px_rgba(74,240,255,0.08)] transition hover:border-cyan-200/30 hover:bg-cyan-300/15 md:inline-block"
          >
            Try ADAM
          </Link>

          {!loading && user ? (
            <div className="hidden items-center gap-2 md:flex">
              {user.photoURL && (
                <Image
                  src={user.photoURL}
                  alt={user.displayName ?? 'User'}
                  width={28}
                  height={28}
                  className="rounded-full"
                />
              )}
              <button
                onClick={signOut}
                className="text-xs text-white/52 transition hover:text-white"
              >
                Sign out
              </button>
            </div>
          ) : (
            <Link
              href="/contact"
              className="hidden rounded-full border border-white/10 px-4 py-2 text-sm font-semibold text-white/78 transition hover:border-white/22 hover:text-white md:inline-block"
            >
              Get a Quote
            </Link>
          )}
        </div>
      </nav>
    </header>
  );
}
