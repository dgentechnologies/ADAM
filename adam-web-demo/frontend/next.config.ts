import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        // Google user profile photos (OAuth sign-in avatars)
        protocol: 'https',
        hostname: 'lh3.googleusercontent.com',
        pathname: '/**',
      },
    ],
  },
  async headers() {
    return [
      {
        // Allow this app to be embedded in an iframe on the DGEN website
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: "frame-ancestors 'self' https://dgentechnologies.com https://*.dgentechnologies.com",
          },
          {
            // Override any default X-Frame-Options so the iframe is allowed
            key: 'X-Frame-Options',
            value: 'ALLOWALL',
          },
          {
            // Required for Firebase signInWithPopup: allows the opener window to
            // retain a reference to the Google OAuth popup so Firebase can detect
            // when auth completes. Without this, COOP blocks window.closed polls.
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin-allow-popups',
          },
        ],
      },
    ];
  },
};

export default nextConfig;
