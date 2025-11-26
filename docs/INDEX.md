# 📚 Aria Navigation - Documentation Index

> **Central documentation hub for the Aria Navigation System**
> Last updated: November 25, 2025

## 🎯 Essential Documents (Start Here)

| Document | Description | Priority |
|----------|-------------|----------|
| [**PROJECT_TIMELINE.md**](history/PROJECT_TIMELINE.md) | **Complete 10-iteration development history** | ⭐⭐⭐ |
| [**CHANGELOG.md**](../CHANGELOG.md) | **Version history with all features/fixes** | ⭐⭐⭐ |
| [**TROUBLESHOOTING.md**](guides/TROUBLESHOOTING.md) | **Complete guide to debugging issues** | ⭐⭐⭐ |
| [Setup Guide](setup/SETUP.md) | Detailed installation and configuration | ⭐⭐ |
| [Quick Reference](guides/QUICK_REFERENCE.md) | Common commands and workflows | ⭐⭐ |

---

## 📖 Quick Start

| Document | Description | Audience |
|----------|-------------|----------|
| [README](../README.md) | Project overview, installation, basic usage | Everyone |
| [Setup Guide](setup/SETUP.md) | Detailed installation and configuration | New users |
| [Quick Reference](guides/QUICK_REFERENCE.md) | Common commands and workflows | Daily users |
| [Mock Observer Guide](guides/MOCK_OBSERVER_GUIDE.md) | Testing without Aria hardware | Developers |

---

## 🏗️ Architecture & Design

| Document | Description | Level |
|----------|-------------|-------|
| [**Architecture & Data Flow**](architecture/architecture_document.md) | **Consolidated architecture, timing, pipelines** | ⭐⭐⭐ |
| [Deep Dive (Archive)](archive/architecture/DEEP_DIVE.md) | Full legacy deep-dive reference | 📦 Archive |
| [Data Flow (Archive)](archive/architecture/DATA_FLOW.md) | Frame-by-frame legacy trace | 📦 Archive |
| [Pipeline Details](architecture/pipeline_overview.md) | Vision pipeline breakdown | ⭐⭐ |
| [Audio System](architecture/audio_spatial_summary.md) | Spatial audio architecture | ⭐ |
| [Navigation Audio Flow](architecture/navigation_audio_flow.md) | Audio routing architecture | ⭐ |

### 📊 Diagrams
| Diagram | Purpose | Location |
|---------|---------|----------|
| [Primary Architecture](architecture/architecture.puml) | Main system architecture | ⭐ PRIMARY |
| [Audio Flow](architecture/navigation_audio_flow.puml) | Audio routing diagram | ⭐ PRIMARY |
| [UML Evolution](diagrams/uml/03_*_clean.puml) | Class/package/sequence diagrams | ⭐ LATEST |
| [Project Timeline](diagrams/project/01_evolution_timeline.puml) | Development timeline visualization | Reference |

**→ See [PUML_AUDIT.md](diagrams/PUML_AUDIT.md) for complete diagram inventory**

---

## 📋 User Guides

| Guide | Purpose | Status |
|-------|---------|--------|
| [**Configuration Guide**](guides/CONFIGURATION_GUIDE.md) | **Complete system configuration and tuning** | ⭐⭐⭐ |
| [Quick Reference](guides/QUICK_REFERENCE.md) | Common commands and workflows | ⭐⭐ |
| [Mock Observer Guide](guides/MOCK_OBSERVER_GUIDE.md) | Testing without Aria hardware | ⭐⭐ |
| [Audio Configuration](guides/audio_config.md) | Audio system setup (macOS/Linux) | ⭐ |
| [**Troubleshooting**](guides/TROUBLESHOOTING.md) | **Complete debugging guide** | ⭐⭐⭐ |
| [**Scalability Guide**](guides/SCALABILITY_GUIDE.md) | Resource capacity and scaling | ⭐ |

---

## 🔧 Development

| Document | Purpose |
|----------|---------|
| [Contributing & Development](development/CONTRIBUTING.md) | Workflow, branches, commits, testing |
| [Development (Archive)](archive/development/) | Metodología, guías de problemas (histórico) |

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

### Performance Optimization
| Document | Focus | Achievement |
|----------|-------|-------------|
| [**Performance Optimization Guide**](guides/PERFORMANCE_OPTIMIZATION.md) | **Consolidated tuning (18–22 FPS)** | TensorRT + ONNX + skips |
| [CUDA Optimization Archive](archive/cuda/) | Phase-by-phase historical docs (FASE 1-4) | 3.5 → 18.4 FPS |

---

## 📊 Project History & Timeline

| Document | Coverage | Purpose |
|----------|----------|---------|
| [**PROJECT_TIMELINE.md**](history/PROJECT_TIMELINE.md) | **All 10 iterations** | **Complete development journey** |
| [**CHANGELOG.md**](../CHANGELOG.md) | **v1.0 → v2.0** | **All features, fixes, improvements** |
| [Daily Notes](history/daily_notes.md) | Chronological | Quick session notes |
| [Phase Plans](archive/phases/) | By phase | Historical planning docs |

---

## 🗂️ Reference & Maintenance

| Document | Purpose |
|----------|---------|
| [PUML Audit](diagrams/PUML_AUDIT.md) | Complete diagram inventory (37 files) |
| [Documentation Reorganization](history/REORGANIZATION_SUMMARY.md) | November 2025 restructuring |
| [Archive](archive/README.md) | Deprecated/completed documentation |

---

## 🔍 Finding Information

### I want to...

1. **Understand the project** → Start with [PROJECT_TIMELINE.md](history/PROJECT_TIMELINE.md)
2. **Install the system** → See [Setup Guide](setup/SETUP.md)
3. **Debug an issue** → Check [TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)
4. **See what changed** → Read [CHANGELOG.md](../CHANGELOG.md)
5. **Optimize performance** → Review [Performance Optimization Guide](guides/PERFORMANCE_OPTIMIZATION.md)
6. **Test without hardware** → Use [Mock Observer Guide](guides/MOCK_OBSERVER_GUIDE.md)
7. **Understand architecture** → See [Architecture & Data Flow](architecture/architecture_document.md)
8. **Find a diagram** → Check [PUML Audit](diagrams/PUML_AUDIT.md)

---

## 📈 Documentation Stats

- **Total Files:** 62 Markdown documents (incl. archive)
- **Active Docs:** 37 (25 archived)
- **Total Lines:** ~17,800 (archivo ~10,200)
- **Diagrams:** 37 PlantUML files
- **Categories:** 9 (setup, guides, architecture, development, testing, migration, history, archive, diagrams)
- **Last Major Update:** November 25, 2025 (Arquitectura consolidada + perf + troubleshooting catálogo)

---

## 🎯 Documentation Goals

### Completed ✅
- ✅ Complete documentation reorganization (9 categories)
- ✅ Central INDEX.md hub
- ✅ 10-iteration timeline documented
- ✅ Complete CHANGELOG (v1.0 → v2.0)
- ✅ Comprehensive troubleshooting guide
- ✅ CUDA optimization consolidation
- ✅ PUML diagram audit
- ✅ Missing documentation filled (SETUP, audio_config, testing)

### Future Improvements
- [ ] API reference documentation
- [ ] Video tutorials
- [ ] Interactive diagram viewer
- [ ] Automated documentation tests

---

**Documentation Status:** ✅ Production Ready  
**Coverage:** Comprehensive (100% of major features)  
**Maintenance:** Active  

---

*For questions or suggestions about documentation, see [CONTRIBUTING.md](../CONTRIBUTING.md)*

---

## 🗄️ Archive

Deprecated or superseded documentation organized in [`archive/`](archive/) folder:
- **migration/** - Historical migration plans (superseded by current guides)
- **phases/** - Completed phase planning documents
- **Root files** - Early optimization docs and deprecated guides

See [archive/README.md](archive/README.md) for complete inventory.

---

## 🆘 Getting Help

1. **First time?** → Start with [README](../README.md) and [Setup Guide](setup/SETUP.md)
2. **Development?** → Check [Contributing & Development](development/CONTRIBUTING.md)
3. **Debugging?** → See [TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)
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
