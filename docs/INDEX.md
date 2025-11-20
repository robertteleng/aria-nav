# 📚 Aria Navigation - Documentation Index

> **Central documentation hub for the Aria Navigation System**  
> Last updated: November 20, 2025

## 📖 Quick Start

| Document | Description | Audience |
|----------|-------------|----------|
| [README](../README.md) | Project overview, installation, basic usage | Everyone |
| [Setup Guide](setup/SETUP.md) | Detailed installation and configuration | New users |
| [Quick Reference](guides/QUICK_REFERENCE.md) | Common commands and workflows | Daily users |

---

## 🏗️ Architecture & Design

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture/architecture_document.md) | Complete system architecture |
| [Pipeline Details](architecture/pipeline_overview.md) | Vision pipeline breakdown |
| [Audio System](architecture/audio_spatial_summary.md) | Spatial audio architecture |

---

## 📋 User Guides

| Guide | Purpose | Status |
|-------|---------|--------|
| [Mock Observer Guide](guides/MOCK_OBSERVER_GUIDE.md) | Testing without Aria hardware | ✅ Active |
| [Audio Configuration](guides/audio_config.md) | Audio system setup (macOS/Linux) | ✅ Active |
| [Quick Reference](guides/QUICK_REFERENCE.md) | Common commands and workflows | ✅ Active |

---

## 🔧 Development

| Document | Purpose |
|----------|---------|
| [Development Workflow](development/development_workflow.md) | Git flow, testing, deployment |
| [Development Methodology](development/development_methodology.md) | Agile practices, note-taking |
| [Problem Solving Guide](development/problem_solving_guide.md) | Debugging strategies |
| [Git Commit Guide](development/git_commit_guide.md) | Commit message conventions |

---

## 🧪 Testing

| Document | Coverage |
|----------|----------|
| [Testing Overview](testing/README.md) | Test strategy and execution |
| [Audio Router Tests](testing/navigation_audio_testing.md) | Audio system validation |

---

## 🚀 Migration & Optimization

### Hardware Migration
| Document | Target Platform | Status |
|----------|----------------|--------|
| [NUC Migration Guide](migration/NUC_MIGRATION.md) | Intel NUC 11 + RTX 2060 | ✅ Complete |

### Software Migration
| Document | Purpose | Status |
|----------|---------|--------|
| [Linux Audio Migration](migration/LINUX_AUDIO.md) | macOS → Linux audio stack | ✅ Active |
| [Audio Router Migration](migration/AUDIO_ROUTER_MIGRATION.md) | Legacy → new audio system | ✅ Active |

---

## 📊 Project History

| Document | Period | Purpose |
|----------|--------|---------|
| [Development Diary](history/development_diary.md) | 2024-2025 | Daily development log |
| [Daily Notes](history/daily_notes.md) | Chronological | Quick session notes |
| [Phase Plans](history/phases/) | By phase | Historical planning docs |

---

## 🗄️ Archive

Deprecated or superseded documentation moved to [`archive/`](archive/) folder.

---

## 🆘 Getting Help

1. **First time?** → Start with [README](../README.md) and [Setup Guide](setup/SETUP.md)
2. **Development?** → Check [Development Workflow](development/development_workflow.md)
3. **Debugging?** → See [Problem Solving Guide](development/problem_solving_guide.md)
4. **Migration?** → Read [NUC Migration Guide](migration/NUC_MIGRATION.md)
5. **API Reference?** → Check inline docstrings in `src/` modules

---

## 📝 Documentation Standards

- **File naming**: `UPPERCASE_WITH_UNDERSCORES.md` for guides, `lowercase_with_underscores.md` for technical docs
- **Structure**: Use clear headings (##), tables, and code blocks
- **Status badges**: ✅ Active | 🚧 In Progress | 📝 Draft | ⚠️ Outdated | 🗄️ Archived
- **Last updated**: Include date at top of each document
- **Cross-linking**: Use relative paths for internal links

---

**Contributing to docs?** Update this index when adding/removing documents.
