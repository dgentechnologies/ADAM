import { Page, Panel } from '@/components/companion-ui';
export default function TermsPage() {
  return (
    <Page title="Terms of use" back="/settings">
      <p className="eyebrow">ADAM COMPANION · OCTOBER 2026</p>
      <h2 className="page-title">A thoughtful companion.</h2>
      <Panel>
        <h3 className="mb-3 font-medium">Using the app</h3>
        <p className="text-fg-muted text-sm leading-7">
          ADAM Companion is provided by DGEN Technologies Pvt. Ltd. Use it with accounts, photos,
          devices, and services you own or have permission to access. Keep access tokens private and
          protect your phone with a screen lock.
        </p>
      </Panel>
      <Panel>
        <h3 className="mb-3 font-medium">Availability</h3>
        <p className="text-fg-muted text-sm leading-7">
          Memories can sync to your account when you choose; photos and notifications stay on your
          phone. ADAM pairing, device status,
          robot controls and laptop controls use a clearly marked simulation with sample devices.
          Demo actions do not control physical hardware. Account sign-in and connected-home services
          require network access and the relevant service configuration. This app does not collect
          payments.
        </p>
      </Panel>
      <Panel>
        <h3 className="mb-3 font-medium">Your data and services</h3>
        <p className="text-fg-muted text-sm leading-7">
          Back up important memories before uninstalling or erasing app data. Third-party services
          may charge for usage under their own terms; this app does not collect payments. Device
          commands should be used responsibly. ADAM is not an emergency, medical, or safety-critical
          system.
        </p>
      </Panel>
      <p className="text-fg-muted text-sm leading-7">
        Your rights under applicable consumer and privacy laws remain unaffected. For support, visit
        the official DGEN Technologies website.
      </p>
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
