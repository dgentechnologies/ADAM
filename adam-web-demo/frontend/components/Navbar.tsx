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
    <header className="sticky top-0 z-50 bg-[#0a0a0a]/80 backdrop-blur-md border-b border-white/10">
      <nav className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-6">
        {/* Logo */}
        <Link href="/" className="shrink-0">
          <Image src="/images/logo.png" alt="DGEN Technologies" width={120} height={40} priority />
        </Link>

        {/* Desktop links */}
        <ul className="hidden lg:flex items-center gap-6 text-sm font-medium text-gray-300">
          {NAV_LINKS.map(({ label, href }) => (
            <li key={href}>
              <Link
                href={href}
                className={`hover:text-white transition ${
                  label === 'ADAM' ? 'text-sky-400 hover:text-sky-300' : ''
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
            className="hidden md:inline-block px-4 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-lg text-sm font-semibold transition"
          >
            Try ADAM
          </Link>

          {!loading && user ? (
            <div className="hidden md:flex items-center gap-2">
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
                className="text-xs text-gray-400 hover:text-white transition"
              >
                Sign out
              </button>
            </div>
          ) : (
            <Link
              href="/contact"
              className="hidden md:inline-block px-4 py-2 border border-white/20 hover:border-white/40 text-white rounded-lg text-sm font-semibold transition"
            >
              Get a Quote
            </Link>
          )}
        </div>
      </nav>
    </header>
  );
}
