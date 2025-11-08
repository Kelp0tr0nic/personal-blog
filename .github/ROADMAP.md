# Roadmap

## 0. Foundation
- --Stabilize theme switcher with new palettes (Catppuccin, Gruvbox, Tokyo Storm) and expose them in UI toggles.--
- Replace remaining AstroPaper references (strings, links) with project branding to avoid confusing visitors.

## 1. Content Experience
- Publish a dedicated About page with timeline, tech stack, and links to external profiles.
- Build a changelog/release notes section sourced from `src/data/blog/_releases` to keep template history accessible.
- Add “Reading lists” or “Topics” hubs that surface posts by tag (AI, protocols, opinions).

## 2. Interactive Demos
- Embed lightweight MCP and A2A visualizations (sequence diagrams or mock message traces) using Astro components.
- Create a playground page that walks through a simple agent workflow (prompt → tool call → response), ideally with toggleable themes.

## 3. Developer Insights
- Introduce “Build Logs” collection: short updates on experiments, including code snippets and learnings.
- Add RSS sub-feeds per tag (`/rss/agents.xml`, `/rss/devlog.xml`) for readers who want focused updates.
- Implement a “Now” page describing current focuses, meetups, and reading list; refresh monthly.

## 4. Automation & Tooling
- Configure GitHub Actions for CI (lint + build) and scheduled nightly build of `preview`.
- Generate OpenGraph images dynamically with Satori for every post, highlighting theme colors.
- Set up search using Pagefind bundle already produced in `build` step; add UI search dialog with keyboard shortcut.
- Explore custom OG card templates: tweak `src/utils/og-templates/post.ts` SVG to reflect brand colors, iconography, and theme variants.

## 5. Engagement & Community
- Add newsletter signup (Resend, Buttondown, or ConvertKit) with double opt-in flow.
- Enable comments via Giscus or a privacy-first alternative; moderate on GitHub Discussions.
- Include “Sponsor / Support” call-to-action aligned with your own funding links.

## 6. Long-Term Experiments
- Build a knowledge base using MCP-style descriptors, powering cross-post content recommendations.
- Publish API or CLI scripts demonstrating MCP integration for other developers.
- Explore multi-language support (i18n) with localized metadata and mirrored post directories.
