import { redirect } from 'next/navigation';

export default function WaitlistPage() {
  // Waitlist form lives on the main DGEN website
  redirect('https://dgentechnologies.com/adam/waitlist');
}
