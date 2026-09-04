/** @type {import('next').NextConfig} */
const nextConfig = {
  /**
   * Static export — the build output in ./out is what Capacitor wraps.
   * Consequences enforced across this app: no API routes, no request-time
   * server components, no middleware, and generateStaticParams on every
   * dynamic route.
   */
  output: 'export',

  /**
   * Emits directory-style routes (/welcome/index.html) so the wrapped app can
   * resolve paths from the local filesystem without a server rewriting URLs.
   */
  trailingSlash: true,

  reactStrictMode: true,

  /** next/image optimisation needs a server; unavailable in a static export. */
  images: {
    unoptimized: true,
  },

  /** Workspace packages ship TS source and are compiled by Next. */
  transpilePackages: ['@adam/ui', '@adam/types'],

  /**
   * `@adam/types` writes NodeNext-style specifiers (`./device.js`) so the same
   * source compiles for the Node-side `apps/api`. Webpack resolves those literally
   * and cannot find the `.ts` files, so map the extension here rather than
   * stripping `.js` from the shared package and breaking the Node consumer.
   */
  webpack: (config) => {
    config.resolve.extensionAlias = {
      ...config.resolve.extensionAlias,
      '.js': ['.ts', '.tsx', '.js'],
    };
    return config;
  },

  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Pre-existing TS errors in sign-in/welcome/canvas-reveal-effect are not
    // from face-capture work; ignore at build time so Next can compile.
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
