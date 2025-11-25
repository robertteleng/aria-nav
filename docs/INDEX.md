# 📚 Aria Navigation - Documentation Index

> **Central documentation hub for the Aria Navigation System**
> Last updated: November 25, 2025

## 🎯 Essential Documents (Start Here)

| Document | Description | Priority |
|----------|-------------|----------|
| [**PROJECT_TIMELINE.md**](PROJECT_TIMELINE.md) | **Complete 10-iteration development history** | ⭐⭐⭐ |
| [**CHANGELOG.md**](../CHANGELOG.md) | **Version history with all features/fixes** | ⭐⭐⭐ |
| [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md) | **Complete guide to debugging issues** | ⭐⭐⭐ |
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

| Document | Description |
|----------|-------------|
| [Architecture Overview](architecture/architecture_document.md) | Complete system architecture |
| [Pipeline Details](architecture/pipeline_overview.md) | Vision pipeline breakdown |
| [Audio System](architecture/audio_spatial_summary.md) | Spatial audio architecture |
| [Navigation Audio Flow](architecture/navigation_audio_flow.md) | Audio routing architecture |

### 📊 Diagrams
| Diagram | Purpose | Location |
|---------|---------|----------|
| [Primary Architecture](architecture/architecture.puml) | Main system architecture | ⭐ PRIMARY |
| [Audio Flow](architecture/navigation_audio_flow.puml) | Audio routing diagram | ⭐ PRIMARY |
| [UML Evolution](diagrams/uml/03_*_clean.puml) | Class/package/sequence diagrams | ⭐ LATEST |
| [Project Timeline](diagrams/project/01_evolution_timeline.puml) | Development timeline visualization | Reference |

**→ See [PUML_AUDIT.md](PUML_AUDIT.md) for complete diagram inventory**

---

## 📋 User Guides

| Guide | Purpose | Status |
|-------|---------|--------|
| [Mock Observer Guide](guides/MOCK_OBSERVER_GUIDE.md) | Testing without Aria hardware | ✅ Active |
| [Audio Configuration](guides/audio_config.md) | Audio system setup (macOS/Linux) | ✅ Active |
| [Quick Reference](guides/QUICK_REFERENCE.md) | Common commands and workflows | ✅ Active |
| [**Troubleshooting**](TROUBLESHOOTING.md) | **Complete debugging guide** | ✅ Active |

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

### Performance Optimization
| Document | Focus | Achievement |
|----------|-------|-------------|
| [**CUDA Optimization**](migration/CUDA_OPTIMIZATION.md) | **Complete optimization guide** | **+426% FPS** |
| [CUDA Phase Documentation](cuda optimization/README.md) | Phase-by-phase details (FASE 1-4) | 3.5 → 18.4 FPS |

---

## 📊 Project History & Timeline

| Document | Coverage | Purpose |
|----------|----------|---------|
| [**PROJECT_TIMELINE.md**](PROJECT_TIMELINE.md) | **All 10 iterations** | **Complete development journey** |
| [**CHANGELOG.md**](../CHANGELOG.md) | **v1.0 → v2.0** | **All features, fixes, improvements** |
| [Development Diary](history/development_diary.md) | 2024-2025 | Daily development log |
| [Daily Notes](history/daily_notes.md) | Chronological | Quick session notes |
| [Phase Plans](history/phases/) | By phase | Historical planning docs |

---

## 🗂️ Reference & Maintenance

| Document | Purpose |
|----------|---------|
| [PUML Audit](PUML_AUDIT.md) | Complete diagram inventory (37 files) |
| [Documentation Reorganization](REORGANIZATION_SUMMARY.md) | November 2025 restructuring |
| [Archive](archive/README.md) | Deprecated/completed documentation |

---

## 🔍 Finding Information

### I want to...

1. **Understand the project** → Start with [PROJECT_TIMELINE.md](PROJECT_TIMELINE.md)
2. **Install the system** → See [Setup Guide](setup/SETUP.md)
3. **Debug an issue** → Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
4. **See what changed** → Read [CHANGELOG.md](../CHANGELOG.md)
5. **Optimize performance** → Review [CUDA Optimization](migration/CUDA_OPTIMIZATION.md)
6. **Test without hardware** → Use [Mock Observer Guide](guides/MOCK_OBSERVER_GUIDE.md)
7. **Understand architecture** → See [Architecture Overview](architecture/architecture_document.md)
8. **Find a diagram** → Check [PUML Audit](PUML_AUDIT.md)

---

## 📈 Documentation Stats

- **Total Files:** 54 Markdown documents
- **Active Docs:** 47 (7 archived)
- **Total Lines:** 17,271 lines of documentation (984KB)
- **Diagrams:** 37 PlantUML files
- **Categories:** 9 (setup, guides, architecture, development, testing, migration, history, archive, diagrams)
- **Archive Structure:** Organized into migration/ and phases/ subdirectories
- **Last Major Update:** November 25, 2025 (Archive reorganization + MLflow integration)

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
