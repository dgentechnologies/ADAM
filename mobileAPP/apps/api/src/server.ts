import fastify from 'fastify';
import cors from '@fastify/cors';
import { wifiRoutes } from './routes/wifi.js';
import { deviceRoutes } from './routes/device.js';
import { memoryRoutes } from './routes/memory.js';
import { creditRoutes } from './routes/credits.js';
import { smartHomeRoutes } from './routes/smart-home.js';

export const app = fastify({ logger: true });

await app.register(cors, {
  origin: true,
});

await app.register(wifiRoutes);
await app.register(deviceRoutes);
await app.register(memoryRoutes);
await app.register(creditRoutes);
await app.register(smartHomeRoutes);


app.get('/health', async () => {
  return { status: 'ok', timestamp: new Date().toISOString() };
});

const start = async () => {
  try {
    const port = Number(process.env.PORT) || 3001;
    await app.listen({ port, host: process.env.HOST || '127.0.0.1' });
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
};

const isTest =
  process.env.NODE_ENV === 'test' ||
  process.env.npm_lifecycle_event === 'test' ||
  process.argv.some((arg) => arg.includes('--test'));

if (!isTest) {
  start();
}
