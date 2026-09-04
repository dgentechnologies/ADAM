import React, { useState } from 'react';
import { History, CheckCircle2, XCircle, Trash2, Filter } from 'lucide-react';
import { ActivityLogItem } from '../types';

interface ActivityLogProps {
  logs: ActivityLogItem[];
  onClearLogs: () => void;
}

export const ActivityLog: React.FC<ActivityLogProps> = ({ logs, onClearLogs }) => {
  const [filter, setFilter] = useState<'all' | 'success' | 'failed'>('all');

  const filteredLogs = logs.filter((item) => {
    if (filter === 'success') return item.status === 'success';
    if (filter === 'failed') return item.status === 'failed';
    return true;
  });

  return (
    <div className="rounded-card bg-charcoal/90 border border-hairline p-6 shadow-subtle">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-white" strokeWidth={1.5} />
          <h3 className="text-base font-bold text-white tracking-tight">Audit & Activity Log</h3>
          <span className="text-xs text-muted font-mono bg-white/5 px-2 py-0.5 rounded-full border border-hairline">
            {logs.length} events
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Filter pills */}
          <div className="inline-flex rounded-full bg-near-black p-1 border border-hairline text-xs">
            <button
              onClick={() => setFilter('all')}
              className={`px-2.5 py-0.5 rounded-full font-medium transition-all ${
                filter === 'all'
                  ? 'bg-white text-black'
                  : 'text-muted hover:text-white'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilter('success')}
              className={`px-2.5 py-0.5 rounded-full font-medium transition-all ${
                filter === 'success'
                  ? 'bg-white text-black'
                  : 'text-muted hover:text-white'
              }`}
            >
              Success
            </button>
            <button
              onClick={() => setFilter('failed')}
              className={`px-2.5 py-0.5 rounded-full font-medium transition-all ${
                filter === 'failed'
                  ? 'bg-white text-black'
                  : 'text-muted hover:text-white'
              }`}
            >
              Failed
            </button>
          </div>

          <button
            onClick={onClearLogs}
            className="p-1.5 rounded-full text-muted hover:text-white hover:bg-surface-container-high transition-colors"
            title="Clear logs"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <p className="text-xs text-muted mb-3">
        Real-time telemetry of remote invocations received from ADAM and local manual test executions.
      </p>

      {filteredLogs.length === 0 ? (
        <div className="rounded-2xl bg-near-black/50 border border-hairline/60 py-10 px-4 text-center">
          <Filter className="w-6 h-6 text-dim mx-auto mb-2" strokeWidth={1.5} />
          <p className="text-xs text-muted">No activity logged in this view</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
          {filteredLogs.map((item) => (
            <div
              key={item.id}
              className="rounded-xl bg-near-black/75 border border-hairline/70 px-3.5 py-2.5 flex items-center justify-between text-xs hover:border-hairline-bright transition-all"
            >
              <div className="flex items-center gap-2.5 min-w-0">
                {item.status === 'success' ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-white shrink-0" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 text-neutral-400 shrink-0" />
                )}
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-medium text-white tracking-tight">
                      {item.actionName}
                    </span>
                    <span className="text-[11px] text-muted truncate">
                      {item.details}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3 shrink-0 pl-3">
                <span className="text-[11px] font-mono text-dim">
                  {item.latencyMs}ms
                </span>
                <span className="text-[11px] font-mono text-muted">
                  {item.timestamp}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
