# Native GUI

## Native GUI structure

The macOS GUI in `gui/` is a generation client backed by the native worker.
`src/app.cplus` registers the Generation and Settings windows, installs the
standard first-responder Edit menu, and starts the app; neither window uses
navigation routes. Each window has its own screen
under `gui/src/screens/`. The generation screen composes the prompt, reference
image, output-option, and preview components from `gui/src/components/`.
Those components build retained `@ui` trees and implement `core::IntoNode` via
`component::child`, so the screen can place them directly in its layout.
Both windows use Facet's `Bar::Blended`. The Generation window follows the
local llama model manager's shell pattern: a full-height split with safe-area
opt-out, a left header that places native `window_buttons()` and supplies a
drag region, and a draggable right header. The compose and preview panes paint
their own backgrounds to the top edge. The compose scroll viewport reaches the
split edge; its content, rather than the viewport, supplies left padding and a
right-side gutter so cards do not sit underneath the overlay scrollbar.
The Settings window remains a simpler standalone screen with the default
safe-area inset.
The main pane is a single flexible canvas rather than a card nested inside
another card. Its header keeps the app identity; the canvas fills the remaining
space and centers one empty-state message until generation finishes. A narrow
footer shows the current phase, denoising step count, and click-to-PNG elapsed
time.
New, Edit, and Save sit at the far right of the header. New is also available
from File → New Image (⌘N). It starts a fresh session by clearing the prompt,
reference strip, current preview, and elapsed/status display while retaining
the chosen model and generation settings. If a generation is active, New
requests cooperative cancellation and ignores its eventual result so an old
image cannot reappear. Generated PNG files already written on disk are not
deleted. Edit and Save stay disabled until a
PNG is published as the current image on the UI thread. The current-image
module then shows it in the canvas and enables both actions: Edit inserts that
file at the front of the reference strip (subject to the ten-image limit), and
Save opens a native save dialog and writes an atomic copy to the chosen path.
The original generated file is not moved or renamed. A successful worker
completion publishes its PNG through this current-image module on the UI thread.

The prompt is an editable text area with no placeholder. It receives focus on
the generation window's first activation, but later activations do not steal
focus back from another control. On AppKit the text-area node backs an
`NSScrollView`, so the screen's `Active` handler focuses its inner `NSTextView`;
nested child components do not receive `Active`. Create starts one request on
the long-running worker; it disables while busy and reveals Cancel beside the
heading. Width and
height are digit-only text fields with live guidance for the current minimum
and multiple-of-32 rule. The seed field also filters typed or pasted input to
ASCII digits. It shares a row with a checkbox labeled Random, which disables
manual entry without erasing its value. Each generation with Random checked
draws a new unsigned 64-bit seed.
Steps use a native picker containing the supported 3, 4, 6, 8, 25, and
40-step choices. The compact + button sits beside the reference-image count;
the thumbnail strip is hidden when empty. A new image appears first and the
strip returns to the left so it is immediately visible. Each local-file
thumbnail fills its card, with a white × on a dark circular backing at the
top-left. The backing provides contrast against pale images, where a shadow
alone was insufficient. At ten images the + button is disabled. The first
reference added to an empty strip reads its EXIF-corrected displayed size and
sets the output to the nearest supported 32-pixel-aligned dimensions. Images
larger than the conditioned path's 1 MP budget are proportionally reduced;
the size hint shows source and selected dimensions and explains the cap. Later
references do not change a manually adjusted size. This is a starting value,
not a locked aspect ratio. The Settings window opens from the native menu and contains
persistent cache mode (Model default, Off, TaylorSeer, or Cache-DiT), cache
threshold (Recommended, 0.12, 0.14, 0.16, or 0.24), and output PNG folder
controls. Recommended resolves to 0.24 for TaylorSeer and 0.16 for Cache-DiT.
The output folder uses `/tmp` when none has been chosen, and the window has a
button to restore that default. These settings feed the next generation
request. Width, height, steps, Random mode, and the last valid manual seed persist
through macOS UserDefaults in the stable
`dev.netdur.qwen-image-cplus.preferences` domain. The GUI accepts only supported
steps, dimensions of at least 256 in multiples of 32, and decimal seeds within
the unsigned 64-bit range when loading or saving. The manual seed is stored as a
string to preserve that full range; incomplete or invalid edits never replace
the last valid saved value. The Model card opens a native file picker, displays
the selected file name, and saves its full path in the same preferences domain;
the worker loads it when a request starts. Selection requires the tokenizer, four
text-encoder shards, and VAE file in subdirectories beside the QIPACK; an
incomplete saved selection shows a warning. The Reference Images and Model
cards also accept Finder file drops anywhere on their surfaces. Both routes
validate real local files: references accept common image extensions and stop
at ten, while the model card accepts a `.qipack` only when those supporting
files are present beside it, and then persists its path. A multi-image drop
keeps the first accepted files in Finder order at the front of the thumbnail
strip; its first image sets the suggested output dimensions. Unsupported files
are ignored. Facet's `allow_file_drop` gesture supplies file paths to the
cards, while their click controls remain available. Prompt text and reference
images remain session-only.

The generation worker in `qwen_image/src/generation_worker.cplus` is driven by
`gui/src/generation_session.cplus`. A single long-lived thread accepts
an owned request through a typed channel, executes the native API, and sends
typed progress and completion events back through another channel. The caller
passes a monotonic timestamp captured at the Create click to `submit`; the
completion event reports elapsed milliseconds through PNG completion. The
service accepts one generation at a time, and `cancel(request_id)` marks only
that request. The observer checks the mark at phase boundaries and after each
denoising step, so cancellation is cooperative and may take up to the current
model operation to finish. It does not interrupt a Metal command buffer, VAE
decode, or PNG write mid-operation. Dropping the service closes its command
channel; it does not forcibly terminate an in-flight request. The GUI polls
worker events on the main thread and retains the previous image after a
cancelled or failed run. A 512x512 three-step GUI smoke produced a PNG in 14
seconds; a second run cancelled in nine seconds with the first image still
visible.
Failures now carry a diagnostic event from the native pipeline through the
worker to the GUI. The preview keeps the short status in its footer and shows
the detailed reason on a separate wrapping line, even when a previous image
remains visible. Validation, reference loading, tokenizer/text, vision/VAE,
transformer, and PNG-output failures report the failing stage and a useful
check; deeper Metal/model diagnostics remain in the app log. Starting another
generation or choosing New clears the old reason, and cancellation does not
show a failure message.

The native API's optional observer is passed through text-to-image and
multi-image paths. It reports preparation, reference/text encoding, denoising
step counts, VAE decoding, and PNG writing, and can return `Cancelled` without
breaking the existing synchronous CLI or C ABI. The worker uses no Facet or
AppKit calls; the GUI drains events on the main thread and stages state updates
there. Mock-worker tests cover progress, elapsed time,
per-request cancellation, and a subsequent request without running inference.
