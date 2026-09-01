'use client';

import { NotYetDesigned, Screen } from '@adam/ui';

import { AppBar } from '@/components/app-bar';

export default function VoiceSettingsPage() {
  return (
    <>
      <AppBar title="Voice & Wake Word" back="/settings" />
      <Screen chrome="both">
        <NotYetDesigned
          title="Voice & Wake Word"
          purpose="How ADAM sounds and what he answers to. Undesigned in the Stitch export."
          bullets={[
            'Wake word choice and sensitivity',
            'Voice selection and speaking rate',
            'Mic mute schedule (quiet hours)',
            'Push-to-talk as an alternative to always-listening',
          ]}
        />
      </Screen>
    </>
  );
}
