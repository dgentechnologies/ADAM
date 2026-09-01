'use client';

import { Button, IconButton, Screen, ScreenHeader, TextField } from '@adam/ui';
import { zodResolver } from '@hookform/resolvers/zod';
import { Eye, EyeOff } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';

import { useSetupStore } from '@/stores/setup-store';

/**
 * `wi_fi_password_entry`.
 *
 * Stitch drew a full fake QWERTY keyboard here. That is dropped deliberately: the
 * OS keyboard is the correct control on both platforms, and a hand-drawn one would
 * break autofill, password managers and every non-Latin layout.
 *
 * WPA2 allows 8–63 characters; the schema enforces it so the CTA cannot submit a
 * value the unit will certainly reject.
 */
const schema = z.object({
  password: z.string().min(8, 'Wi-Fi passwords are at least 8 characters.').max(63),
});

type FormValues = z.infer<typeof schema>;

export default function WifiPasswordPage() {
  const router = useRouter();
  const ssid = useSetupStore((state) => state.selectedSsid);
  const complete = useSetupStore((state) => state.complete);
  const [revealed, setRevealed] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    mode: 'onChange',
    defaultValues: { password: '' },
  });

  const onSubmit = handleSubmit(() => {
    // The password is handed to the unit over the local channel, never stored.
    complete('wifi-password');
    router.push('/connecting');
  });

  return (
    <Screen className="pt-stack-md">
      <form onSubmit={onSubmit} className="flex flex-1 flex-col">
        <ScreenHeader size="sm" title={`Enter password for ‘${ssid ?? 'your network'}’`} />

        {/* Stitch puts the field and its CTA together as one block directly under
            the headline rather than pinning the button to the bottom rail. */}
        <div className="flex flex-col gap-stack-md pt-stack-lg">
          <TextField
            variant="underline"
            placeholder="Wi-Fi password"
            aria-label="Wi-Fi password"
            type={revealed ? 'text' : 'password'}
            autoComplete="current-password"
            autoFocus
            error={errors.password?.message}
            trailing={
              <IconButton
                size="sm"
                variant="ghost"
                aria-label={revealed ? 'Hide password' : 'Show password'}
                onClick={() => setRevealed((value) => !value)}
              >
                {revealed ? (
                  <EyeOff className="h-5 w-5" strokeWidth={1.5} />
                ) : (
                  <Eye className="h-5 w-5" strokeWidth={1.5} />
                )}
              </IconButton>
            }
            {...register('password')}
          />

          <Button type="submit" block variant="primary" disabled={!isValid}>
            Connect
          </Button>
        </div>
      </form>
    </Screen>
  );
}
