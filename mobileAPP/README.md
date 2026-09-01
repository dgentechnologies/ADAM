# ADAM Companion App

Companion mobile application and API for the ADAM AI Hardware Robot by **DGEN Technologies Pvt. Ltd.**

## Architecture

This project is a Turborepo monorepo managed with pnpm workspaces.

`
mobileAPP/
├── apps/
│   ├── web/            # Next.js 15 (React 19, Tailwind, Framer Motion) — Static Export
│   ├── api/            # Fastify + Node.js backend API
│   └── mobile-shell/   # Capacitor shell wrapping web into Android / iOS APK
├── packages/
│   ├── ui/             # Shared React UI design system (Monochrome / dark aesthetic)
│   ├── types/          # Shared TypeScript interfaces & Zod validation contracts
│   └── config/         # Shared Tailwind, TypeScript & ESLint configurations
├── docs/               # Technical build specs, design guidelines, and setup flows
└── ref/                # Reference Stitch UI screens & component design assets
`

## Getting Started

### Prerequisites
- Node.js >= 20.11.0
- pnpm >= 9.x

### Available Scripts

- pnpm dev: Start all apps in development mode concurrently
- pnpm dev:web: Start the Next.js web companion app
- pnpm dev:api: Start the Fastify backend API
- pnpm build: Build all workspaces (compiles static export in pps/web/out)
- pnpm typecheck: Run TypeScript type-checking across all packages
- pnpm cap:sync: Sync the static web build into the Capacitor mobile shell
- pnpm cap:android: Open the Android project in Android Studio for APK generation
