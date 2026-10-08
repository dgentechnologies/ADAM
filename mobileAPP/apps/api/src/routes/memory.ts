import type { FastifyInstance, FastifyPluginAsync } from 'fastify';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Root ADAM folder is 3 levels up from apps/api/src/routes
const ADAM_ROOT = path.resolve(__dirname, '../../../../../');
const ADAM_MEMORY_PATH = path.join(ADAM_ROOT, 'adam_memory.json');

interface MemoryItem {
  id: string;
  deviceId: string;
  kind: 'person' | 'fact';
  label: string;
  content: string;
  hasFaceProfile: boolean;
  photoUrl?: string;
  createdAt: string;
}

interface FaceItem {
  id: string;
  name: string;
  role: string;
  photoDataUrl: string;
  views?: {
    front?: string;
    left?: string;
    right?: string;
  };
  capturedAt: string;
}

const customMemories: MemoryItem[] = [];
const registeredFaces: FaceItem[] = [];

// Helper to load facts from root adam_memory.json
function loadAdamRootMemories(): MemoryItem[] {
  const items: MemoryItem[] = [];
  const deviceId = '7c9e6679-7425-40de-944b-e07fc1f90ae7';

  try {
    if (fs.existsSync(ADAM_MEMORY_PATH)) {
      const raw = fs.readFileSync(ADAM_MEMORY_PATH, 'utf-8');
      const data = JSON.parse(raw);

      if (data.user_realname) {
        items.push({
          id: 'mem-person-owner',
          deviceId,
          kind: 'person',
          label: data.user_realname,
          content: `${data.user_realname} (Owner)`,
          hasFaceProfile: true,
          createdAt: '2026-08-01T09:00:00.000Z',
        });
      }

      if (data.favorite_pizza) {
        items.push({
          id: 'mem-fact-pizza',
          deviceId,
          kind: 'fact',
          label: 'Favorites',
          content: `Favorite pizza: ${data.favorite_pizza}`,
          hasFaceProfile: false,
          createdAt: '2026-08-15T12:00:00.000Z',
        });
      }

      if (data.last_user_interaction) {
        items.push({
          id: 'mem-fact-interaction',
          deviceId,
          kind: 'fact',
          label: 'Interaction',
          content: data.last_user_interaction,
          hasFaceProfile: false,
          createdAt: '2026-08-28T16:30:00.000Z',
        });
      }

      if (data.user_last_context) {
        items.push({
          id: 'mem-fact-context',
          deviceId,
          kind: 'fact',
          label: 'Context',
          content: data.user_last_context,
          hasFaceProfile: false,
          createdAt: '2026-08-29T18:00:00.000Z',
        });
      }
    }
  } catch (err) {
    console.warn('[memory] Could not read root adam_memory.json:', err);
  }

  return items;
}

export const memoryRoutes: FastifyPluginAsync = async (fastify: FastifyInstance) => {
  // Get all memory entries
  fastify.get('/api/memory', async () => {
    const rootMemories = loadAdamRootMemories();
    // Combine root memories, custom memories, and faces converted to person memories
    const all = [...rootMemories, ...customMemories];

    // Add registered faces as people if not already present
    for (const face of registeredFaces) {
      if (!all.some((m) => m.kind === 'person' && m.label.toLowerCase() === face.name.toLowerCase())) {
        all.unshift({
          id: `mem-face-${face.id}`,
          deviceId: '7c9e6679-7425-40de-944b-e07fc1f90ae7',
          kind: 'person',
          label: face.name,
          content: `${face.name} · Face ID active`,
          hasFaceProfile: true,
          photoUrl: face.photoDataUrl,
          createdAt: face.capturedAt,
        });
      }
    }

    return all;
  });

  // Add new memory item
  fastify.post<{ Body: Partial<MemoryItem> }>('/api/memory', async (request, reply) => {
    const body = request.body || {};
    if (!body.content) {
      reply.status(400);
      return { error: 'Memory content is required' };
    }

    const newItem: MemoryItem = {
      id: `mem-${Date.now()}`,
      deviceId: '7c9e6679-7425-40de-944b-e07fc1f90ae7',
      kind: body.kind || 'fact',
      label: body.label || (body.kind === 'person' ? 'Person' : 'General'),
      content: body.content,
      hasFaceProfile: !!body.hasFaceProfile,
      createdAt: new Date().toISOString(),
    };

    customMemories.push(newItem);
    return newItem;
  });

  // Delete memory item
  fastify.delete<{ Params: { id: string } }>('/api/memory/:id', async (request) => {
    const { id } = request.params;
    const index = customMemories.findIndex((m) => m.id === id);
    if (index !== -1) {
      customMemories.splice(index, 1);
    }
    return { success: true, id };
  });

  // Get registered faces
  fastify.get('/api/faces', async () => {
    // If no faces registered yet, check root adam_memory.json for user_realname
    if (registeredFaces.length === 0) {
      try {
        if (fs.existsSync(ADAM_MEMORY_PATH)) {
          const raw = fs.readFileSync(ADAM_MEMORY_PATH, 'utf-8');
          const data = JSON.parse(raw);
          if (data.user_realname) {
            return [
              {
                id: 'face-owner',
                name: data.user_realname,
                role: 'Owner',
                photoDataUrl: '',
                capturedAt: '2026-08-01T09:00:00.000Z',
              },
            ];
          }
        }
      } catch {
        // ignore
      }
    }
    return registeredFaces;
  });

  // Save captured face photo
  fastify.post<{
    Body: {
      name: string;
      photoDataUrl: string;
      views?: { front?: string; left?: string; right?: string };
      role?: string;
    };
  }>('/api/faces', async (request, reply) => {
    const body = request.body || {};
    if (!body.photoDataUrl && !body.name) {
      reply.status(400);
      return { error: 'photoDataUrl or name is required' };
    }

    const faceItem: FaceItem = {
      id: `face-${Date.now()}`,
      name: body.name || 'Owner',
      role: body.role || 'Owner',
      photoDataUrl: body.photoDataUrl || '',
      views: body.views,
      capturedAt: new Date().toISOString(),
    };

    // Keep primary owner profile updated
    const existingIndex = registeredFaces.findIndex(
      (f) => f.role === 'Owner' || f.name.toLowerCase() === faceItem.name.toLowerCase(),
    );
    if (existingIndex !== -1) {
      registeredFaces[existingIndex] = faceItem;
    } else {
      registeredFaces.unshift(faceItem);
    }

    return { success: true, face: faceItem };
  });

  // Delete face
  fastify.delete<{ Params: { id: string } }>('/api/faces/:id', async (request) => {
    const { id } = request.params;
    const index = registeredFaces.findIndex((f) => f.id === id);
    if (index !== -1) {
      registeredFaces.splice(index, 1);
    }
    return { success: true, id };
  });
};
