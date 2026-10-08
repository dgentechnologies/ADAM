'use client';

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-black p-4 text-white">
      <h2 className="text-lg font-semibold">Something went wrong</h2>
      <p className="mt-2 text-sm text-white/50">
        {error?.message || 'An unexpected error occurred.'}
      </p>
      <button
        onClick={() => reset()}
        className="mt-4 rounded-full bg-white px-5 py-2 text-sm font-medium text-black"
      >
        Try again
      </button>
    </div>
  );
}
