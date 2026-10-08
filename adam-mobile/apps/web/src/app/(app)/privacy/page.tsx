import { Page, Panel } from '@/components/companion-ui';
export default function PrivacyPage() {
  return (
    <Page title="Privacy notice" back="/settings">
      <p className="eyebrow">ADAM COMPANION · OCTOBER 2026</p>
      <h2 className="page-title">
        Your information,
        <br />
        handled with care.
      </h2>
      <Panel>
        <h3 className="mb-3 font-medium">On this phone</h3>
        <p className="text-fg-muted text-sm leading-7">
          Memories, gallery photos, face images, and preferences are stored locally in the app.
          Memories and selected preferences can also sync to your account when you enable account
          sync and tap Sync now. Photos, face images, and private keys stay on this phone.
          Uninstalling removes local data. Backups and photo sharing happen only when you choose them.
        </p>
      </Panel>
      <Panel>
        <h3 className="mb-3 font-medium">If you sign in</h3>
        <p className="text-fg-muted text-sm leading-7">
          Google Firebase handles authentication. Your account ID, name, email, and optional Google
          avatar are used for your cloud profile. In Your profile, you can choose to share memories,
          people, voice, wake word, and AI selections with the same account on desktop. Turning sync
          off keeps existing phone and account data; deleting your account removes its shared data.
          On Android, sign-in credentials and saved API
          tokens use encrypted storage backed by Android Keystore. You can sign out or delete your
          account in Your profile.
        </p>
      </Panel>
      <Panel>
        <h3 className="mb-3 font-medium">Services you connect</h3>
        <p className="text-fg-muted text-sm leading-7">
          Home Assistant requests go to the server you configure and include its access token.
          Verifying a Gemini key sends that key directly to Google. External services have their own
          privacy terms. The app does not silently discover devices, collect location, or run
          background microphone recording.
        </p>
      </Panel>
      <Panel>
        <h3 className="mb-3 font-medium">Notifications, with your permission</h3>
        <p className="text-fg-muted text-sm leading-7">
          Notification access is requested only when you tap Enable in Notifications. Android asks
          you to approve access in its system screen. Once enabled, ADAM can read notification
          titles, messages, app names, and times while in the background. The latest 100 are kept in
          this app’s private storage, excluded from backups, and never uploaded or sent to your
          robot. Android may redact sensitive content. Pause reading or clear history in
          Notifications, and revoke access in Android settings at any time.
        </p>
      </Panel>
      <Panel>
        <h3 className="mb-3 font-medium">Your choices</h3>
        <p className="text-fg-muted text-sm leading-7">
          Camera access is optional and requested when needed. Delete photos and face profiles from
          their screens. Export, restore, or erase local data in Settings → Your data. Visit DGEN
          Technologies’ official website for support and privacy requests.
        </p>
      </Panel>
      <a
        className="py-3 text-sm underline"
        href="https://dgentechnologies.com"
        target="_blank"
        rel="noopener noreferrer"
      >
        DGEN Technologies
      </a>
    </Page>
  );
}
