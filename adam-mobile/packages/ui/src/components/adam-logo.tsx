import { cn } from '../lib/cn';

/** The desktop release logo; animated robot expressions remain AdamFaceMark. */
export function AdamLogo({ size = 64, className }: { size?: number; className?: string }) {
  return <img src="/assets/adam-logo.png" alt="ADAM" width={size} height={size} className={cn('shrink-0 object-contain', className)} />;
}
