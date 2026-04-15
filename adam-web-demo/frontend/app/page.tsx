import Link from 'next/link';
import Image from 'next/image';

export default function HomePage() {
  return (
    <section className="flex flex-col items-center justify-center min-h-[80vh] px-6 text-center">
      <Image
        src="/images/logo.png"
        alt="DGEN Technologies"
        width={160}
        height={60}
        className="mb-8"
        priority
      />
      <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-4">
        Innovate.{' '}
        <span className="text-sky-400">Integrate.</span>{' '}
        Inspire.
      </h1>
      <p className="text-lg md:text-xl text-gray-400 max-w-2xl mb-8">
        DGEN Technologies builds smart city infrastructure, IoT systems, and AI products.
        Made in India — headquartered in Kolkata.
      </p>

      {/* ADAM teaser */}
      <div className="border border-sky-500/30 rounded-2xl p-8 max-w-xl w-full bg-sky-950/20 mb-8">
        <Image
          src="/images/adam-desktop-ai-module.png"
          alt="ADAM — Autonomous Desktop AI Module"
          width={200}
          height={200}
          className="mx-auto mb-4 rounded-xl"
        />
        <p className="text-sky-400 font-semibold text-sm uppercase tracking-widest mb-2">
          Something Big is Cooking
        </p>
        <h2 className="text-2xl font-bold mb-2">ADAM is Coming Soon</h2>
        <p className="text-gray-400 text-sm mb-4">
          Autonomous Desktop AI Module — your desk, upgraded.
        </p>
        <div className="flex gap-3 justify-center">
          <Link
            href="/adam"
            className="px-5 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-lg text-sm font-semibold transition"
          >
            Learn More
          </Link>
          <Link
            href="/adam/waitlist"
            className="px-5 py-2 border border-sky-500 hover:bg-sky-500/10 text-sky-400 rounded-lg text-sm font-semibold transition"
          >
            Join Waitlist
          </Link>
        </div>
      </div>

      <Link
        href="/services"
        className="text-gray-400 hover:text-white text-sm underline underline-offset-4 transition"
      >
        Explore our services →
      </Link>
    </section>
  );
}
