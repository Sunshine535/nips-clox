# Framework Diagram Prompt

**Paper**: CLOX: Partial Masking as a Control for Synthetic Reasoning

## Image Generation Prompt

Create a clean academic methodology framework diagram for a top-tier ML conference paper, titled “CLOX: Partial Masking as a Control for Synthetic Reasoning.” Use a white or very light gray background (#F7F7F7), flat vector style, subtle soft shadows, rounded-rectangle modules, thin consistent outlines, and a harmonious muted palette: blue #4477AA, teal #44AA99, warm gold #CCBB44, soft purple #AA3377, with gray connectors #666666. Use a left-to-right data flow with evenly spaced aligned blocks and minimal text.

Main pipeline: (1) “Task Input / Query” box on the far left in blue. Arrow to (2) “CLOX Controller” in teal, with small sublabel “mask ratio, stage, policy.” From this controller, branch to three vertically stacked intervention modules in soft purple: “Prompt / Inference Masking,” “Reasoning-Step Masking,” and “MoE Activation Masking.” Each branch should visually indicate partial token/slot masking with small blanked segments or dotted placeholders. These three feed into a central larger module in blue-teal: “Masked Reasoning Engine,” annotated with “cloze-style completion.” Arrow to (4) “Candidate Completions” in gold, then to (5) “Scoring / Consistency Check” in teal, then to (6) “Final Answer” in blue on the right.

Add a thin feedback arrow from “Scoring / Consistency Check” back to “CLOX Controller” labeled “adaptive masking.” Include a small lower auxiliary strip showing “Base LLM” beneath the intervention and reasoning modules, indicating shared backbone. Typography should be modern sans-serif, minimal labels only, crisp and publication-ready, high information density but uncluttered. No photorealism, no 3D, no decorative icons beyond simple token-mask glyphs and arrows.

## Usage Instructions

1. Copy the prompt above into an AI image generator (DALL-E 3, Midjourney, Ideogram, etc.)
2. Generate the image at high resolution (2048x1024 or similar landscape)
3. Save as `framework_diagram.png` in the same `charts/` folder
4. Insert into the paper's Method section using:
   - LaTeX: `\includegraphics[width=\textwidth]{charts/framework_diagram.png}`
   - Markdown: `![Framework Overview](charts/framework_diagram.png)`
