'use client';

import { cn } from '@adam/ui';
import { Brain, Home, Images, LampCeiling, Settings } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

/**
 * Five-tab bar: Home / Gallery / Smart Home / Memory / Settings.
 *
 * The Stitch export contained three different tab sets (one with
 * Status/Commands, one three-tab Home/Chat/Settings, one with a mic FAB) and even
 * put a tab bar on setup screens. Both the product spec and the technical spec
 * agree on the set below, and that setup has no tab bar — so the specs win here.
 */
const TABS = [
  { href: '/home', label: 'Home', Icon: Home },
  { href: '/gallery', label: 'Gallery', Icon: Images },
  // "Smart Home" wraps onto two lines at 360–390px and collides with its
  // neighbours, so the tab shows the short form and the full name stays in the
  // accessible name.
  { href: '/smart-home', label: 'Smart', fullLabel: 'Smart Home', Icon: LampCeiling },
  { href: '/memory', label: 'Memory', Icon: Brain },
  { href: '/settings', label: 'Settings', Icon: Settings },
] as const;

export function TabBar() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Primary"
      className="chrome-blur fixed inset-x-0 bottom-0 z-50 pb-safe shadow-[0_-1px_8px_rgba(0,0,0,0.4)]"
    >
      <ul className="flex h-tabbar-h items-center justify-between px-stack-sm">
        {TABS.map((tab) => {
          const { href, label, Icon } = tab;
          const fullLabel = 'fullLabel' in tab ? tab.fullLabel : label;
          // Settings has sub-routes, so match on prefix rather than equality.
          const active = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <li key={href} className="min-w-0 flex-1">
              <Link
                href={href}
                aria-label={fullLabel}
                aria-current={active ? 'page' : undefined}
                className={cn(
                  'flex flex-col items-center justify-center gap-1 py-2',
                  'transition-opacity duration-base ease-standard',
                  active ? 'text-fg opacity-100' : 'text-fg-muted opacity-60',
                )}
              >
                <Icon className="h-6 w-6" strokeWidth={1.5} aria-hidden />
                <span className="w-full truncate text-center text-label-xs uppercase">{label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
