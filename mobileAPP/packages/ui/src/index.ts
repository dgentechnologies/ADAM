/**
 * @adam/ui public surface.
 *
 * Everything here is presentational and framework-agnostic — no next/link, no
 * next/navigation. Routing-aware chrome (app bar, tab bar, wizard shell) lives in
 * apps/web because it needs the router, and duplicating that into the library
 * would tie the component set to Next.
 */
export { cn } from './lib/cn';

export { AdamFaceMark } from './components/adam-face-mark';
export type { AdamFaceMarkProps, FaceExpression, FaceSize } from './components/adam-face-mark';

export { Button, IconButton, buttonVariants } from './components/button';
export type { ButtonProps, IconButtonProps } from './components/button';

export { Card, CardGroup } from './components/card';
export type { CardProps } from './components/card';

export { EmptyState, NotYetDesigned } from './components/empty-state';
export type { EmptyStateProps } from './components/empty-state';

export { ListRow } from './components/list-row';
export type { ListRowProps } from './components/list-row';

export { OptionCard } from './components/option-card';
export type { OptionCardProps } from './components/option-card';

export { ProgressTrack, RadarSweep } from './components/radar-sweep';
export type { RadarSweepProps } from './components/radar-sweep';

export { Screen, ScreenActions, ScreenHeader } from './components/screen';

export { SegmentedControl } from './components/segmented-control';
export type { SegmentedControlProps } from './components/segmented-control';

export { StatusDot } from './components/status-dot';
export type { DeviceStatusKind, StatusDotProps } from './components/status-dot';

export { StepChecklist, StepProgress } from './components/step-progress';
export type { ChecklistItem, ChecklistState, StepProgressProps } from './components/step-progress';

export { TextField, DisplayField } from './components/text-field';
export type { TextFieldProps } from './components/text-field';

export { Toggle } from './components/toggle';
export type { ToggleProps } from './components/toggle';

export { Wordmark } from './components/wordmark';
