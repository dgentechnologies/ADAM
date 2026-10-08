import type { FastifyInstance, FastifyPluginAsync } from 'fastify';

export const creditRoutes: FastifyPluginAsync = async (fastify: FastifyInstance) => {
  fastify.get('/api/credits/balance', async () => {
    return {
      deviceId: '7c9e6679-7425-40de-944b-e07fc1f90ae7',
      remainingMinutes: 380,
      totalPurchasedMinutes: 400,
      updatedAt: new Date().toISOString(),
    };
  });

  fastify.get('/api/credits/packs', async () => {
    return [
      { id: 'trial', name: 'Trial', pricePaise: 59900, priceLabel: '₹599', estimatedMinutes: 60, isMostPopular: false },
      { id: 'starter', name: 'Starter', pricePaise: 149900, priceLabel: '₹1,499', estimatedMinutes: 180, isMostPopular: false },
      { id: 'standard', name: 'Standard', pricePaise: 299900, priceLabel: '₹2,999', estimatedMinutes: 400, isMostPopular: true },
      { id: 'value', name: 'Value', pricePaise: 549900, priceLabel: '₹5,499', estimatedMinutes: 850, isMostPopular: false },
      { id: 'pro', name: 'Pro', pricePaise: 1199900, priceLabel: '₹11,999', estimatedMinutes: 2000, isMostPopular: false },
    ];
  });
};
