import React, { useState } from 'react';
import {
  Volume2,
  VolumeX,
  Sun,
  Lock,
  PlaySquare,
  Camera,
  Clipboard,
  Sliders,
  Check,
  AlertCircle
} from 'lucide-react';
import { AdamAction } from '../types';

interface ActionControlsProps {
  actions: AdamAction[];
  onToggleAction: (actionId: string) => void;
  onRunTest: (actionId: string, param?: string | number) => void;
  controlPaused: boolean;
}

export const ActionControls: React.FC<ActionControlsProps> = ({
  actions,
  onToggleAction,
  onRunTest,
  controlPaused,
}) => {
  const [testVolume, setTestVolume] = useState<number>(65);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [testBrightness, setTestBrightness] = useState<number>(80);
  const [feedbackMsg, setFeedbackMsg] = useState<{ id: string; text: string } | null>(null);

  const getActionIcon = (iconName: string) => {
    const iconProps = { className: 'w-4 h-4 text-white', strokeWidth: 1.5 };
    switch (iconName) {
      case 'volume':
        return <Volume2 {...iconProps} />;
      case 'sun':
        return <Sun {...iconProps} />;
      case 'lock':
        return <Lock {...iconProps} />;
      case 'media':
        return <PlaySquare {...iconProps} />;
      case 'camera':
        return <Camera {...iconProps} />;
      case 'clipboard':
        return <Clipboard {...iconProps} />;
      default:
        return <Sliders {...iconProps} />;
    }
  };

  const handleTestTrigger = (actionId: string, param?: string | number, label?: string) => {
    onRunTest(actionId, param);
    setFeedbackMsg({
      id: actionId,
      text: label || `Tested ${actionId}${param !== undefined ? ` (${param})` : ''}`,
    });
    setTimeout(() => {
      setFeedbackMsg(null);
    }, 2400);
  };

  return (
    <div className="space-y-6">
      {/* Interactive Manual Test Bench (§3.7 Manual test buttons) */}
      <div className="rounded-card bg-charcoal/90 border border-hairline p-6 shadow-subtle">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-white" strokeWidth={1.5} />
            <h3 className="text-base font-bold text-white tracking-tight">Interactive Action Playground</h3>
          </div>
          <span className="text-xs text-muted">Test local actions without asking ADAM</span>
        </div>

        {controlPaused && (
          <div className="mb-4 flex items-center gap-2 px-3.5 py-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-200 text-xs">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Laptop control is currently paused. Test commands will simulate execution.</span>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Volume Test Card */}
          <div className="rounded-2xl bg-near-black/70 border border-hairline/80 p-4 flex flex-col justify-between gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-semibold text-white">
                <Volume2 className="w-4 h-4 text-muted" strokeWidth={1.5} />
                <span>Volume Control</span>
              </div>
              <span className="text-xs font-mono text-white bg-white/5 px-2 py-0.5 rounded-full border border-hairline">
                {isMuted ? 'MUTED' : `${testVolume}%`}
              </span>
            </div>

            <div className="space-y-2">
              <input
                type="range"
                min="0"
                max="100"
                value={testVolume}
                disabled={isMuted}
                onChange={(e) => setTestVolume(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex items-center justify-between text-[11px] text-muted">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            <div className="flex items-center gap-2 pt-1">
              <button
                onClick={() =>
                  handleTestTrigger(
                    'volume_set',
                    testVolume,
                    `Volume set to ${testVolume}%`
                  )
                }
                className="btn-primary flex-1 text-xs py-1.5"
              >
                Apply {testVolume}%
              </button>
              <button
                onClick={() => {
                  const nextMuted = !isMuted;
                  setIsMuted(nextMuted);
                  handleTestTrigger(
                    nextMuted ? 'volume_mute' : 'volume_unmute',
                    undefined,
                    nextMuted ? 'System Audio Muted' : 'System Audio Unmuted'
                  );
                }}
                className={`px-3 py-1.5 rounded-full text-xs font-medium border flex items-center gap-1 transition-all ${
                  isMuted
                    ? 'bg-white text-black border-white'
                    : 'bg-surface-container-high text-on-surface border-hairline hover:text-white'
                }`}
              >
                {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
                <span>{isMuted ? 'Unmute' : 'Mute'}</span>
              </button>
            </div>
          </div>

          {/* Brightness Test Card */}
          <div className="rounded-2xl bg-near-black/70 border border-hairline/80 p-4 flex flex-col justify-between gap-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-semibold text-white">
                <Sun className="w-4 h-4 text-muted" strokeWidth={1.5} />
                <span>Display Brightness</span>
              </div>
              <span className="text-xs font-mono text-white bg-white/5 px-2 py-0.5 rounded-full border border-hairline">
                {testBrightness}%
              </span>
            </div>

            <div className="space-y-2">
              <input
                type="range"
                min="0"
                max="100"
                value={testBrightness}
                onChange={(e) => setTestBrightness(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex items-center justify-between text-[11px] text-muted">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            <div className="flex items-center gap-2 pt-1">
              <button
                onClick={() =>
                  handleTestTrigger(
                    'brightness_set',
                    testBrightness,
                    `Brightness set to ${testBrightness}%`
                  )
                }
                className="btn-primary flex-1 text-xs py-1.5"
              >
                Apply {testBrightness}%
              </button>
              <button
                onClick={() => {
                  const half = 50;
                  setTestBrightness(half);
                  handleTestTrigger('brightness_set', half, 'Brightness reset to 50%');
                }}
                className="btn-secondary text-xs py-1.5"
              >
                Reset 50%
              </button>
            </div>
          </div>
        </div>

        {/* Quick Test Actions Strip */}
        <div className="mt-4 pt-4 border-t border-hairline/60 flex flex-wrap items-center justify-between gap-2.5">
          <span className="text-xs text-muted font-medium">Quick Simulated Actions:</span>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => handleTestTrigger('system_lock', undefined, 'Lock screen simulated')}
              className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
            >
              <Lock className="w-3 h-3" />
              <span>Test Lock Screen</span>
            </button>
            <button
              onClick={() => handleTestTrigger('media_toggle', undefined, 'Media play/pause toggled')}
              className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
            >
              <PlaySquare className="w-3 h-3" />
              <span>Test Media Play/Pause</span>
            </button>
            <button
              onClick={() => handleTestTrigger('screen_capture', undefined, 'Screenshot sent to ADAM')}
              className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
            >
              <Camera className="w-3 h-3" />
              <span>Test Screenshot</span>
            </button>
          </div>
        </div>

        {/* Feedback Banner */}
        {feedbackMsg && (
          <div className="mt-3 py-2 px-3 rounded-xl bg-white/10 border border-white/15 text-white text-xs flex items-center gap-2 animate-fadeIn">
            <Check className="w-3.5 h-3.5 text-white" />
            <span>{feedbackMsg.text}</span>
          </div>
        )}
      </div>

      {/* Available Actions Allow-List (§3.7) */}
      <div className="rounded-card bg-charcoal/90 border border-hairline p-6 shadow-subtle">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">Permission & Action Allow-list</h3>
            <p className="text-xs text-muted mt-0.5">
              Choose which system controls ADAM is authorized to perform on this laptop.
            </p>
          </div>
          <span className="text-xs font-mono text-on-surface-variant bg-white/5 border border-hairline px-2.5 py-1 rounded-full">
            {actions.filter((a) => a.enabled).length}/{actions.length} Enabled
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
          {actions.map((action) => (
            <div
              key={action.id}
              className={`rounded-2xl border p-3.5 transition-all flex items-center justify-between ${
                action.enabled
                  ? 'bg-near-black/70 border-hairline/90'
                  : 'bg-near-black/30 border-hairline/40 opacity-60'
              }`}
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-surface-container-high flex items-center justify-center border border-hairline shrink-0">
                  {getActionIcon(action.icon)}
                </div>
                <div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-semibold text-white tracking-tight">
                      {action.name}
                    </span>
                    {action.isDestructive && (
                      <span className="text-[9px] uppercase font-bold text-amber-300/90 bg-amber-500/10 px-1.5 py-0.2 rounded border border-amber-500/20">
                        Confirm
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-muted leading-tight mt-0.5">{action.description}</p>
                </div>
              </div>

              {/* Minimalist Toggle conforming to DESIGN.md */}
              <button
                type="button"
                role="switch"
                aria-checked={action.enabled}
                onClick={() => onToggleAction(action.id)}
                className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors duration-200 ease-in-out border border-hairline ${
                  action.enabled ? 'bg-hairline-bright' : 'bg-surface-container-low'
                }`}
              >
                <span
                  className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow-md transition duration-200 ease-in-out mt-[1px] ${
                    action.enabled ? 'translate-x-[18px]' : 'translate-x-[2px]'
                  }`}
                />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
