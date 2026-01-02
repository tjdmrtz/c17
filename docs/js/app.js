/**
 * Wallpaper Groups Interactive Explorer
 * Main application logic
 * 
 * MATHEMATICAL CORRECTNESS: Patterns are generated with EXACT symmetries
 * by constructing them from asymmetric fundamental domains.
 */

class WallpaperExplorer {
    constructor() {
        this.currentGroup = 'p1';
        this.operations = [];
        this.currentMatrix = SymmetryOps.identity();
        this.originalImageData = null;
        this.transformedImageData = null;
        
        // Track current position in Cayley graph
        this.currentCayleyNode = 'e';
        this.cayleyNodes = [];  // Store node positions for interactivity
        
        // Canvas elements
        this.originalCanvas = document.getElementById('originalCanvas');
        this.transformedCanvas = document.getElementById('transformedCanvas');
        this.differenceCanvas = document.getElementById('differenceCanvas');
        
        this.originalCtx = this.originalCanvas.getContext('2d');
        this.transformedCtx = this.transformedCanvas.getContext('2d');
        this.differenceCtx = this.differenceCanvas.getContext('2d');
        
        // Canvas size
        this.canvasSize = 300;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.selectGroup('p1');
    }
    
    setupEventListeners() {
        // Navigation buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const group = e.currentTarget.dataset.group;
                this.selectGroup(group);
            });
        });
        
        // Symmetry operation buttons
        document.querySelectorAll('.symmetry-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const operation = e.currentTarget.dataset.operation;
                const params = { ...e.currentTarget.dataset };
                delete params.operation;
                this.applyOperation(operation, params);
            });
        });
        
        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.reset();
        });
        
        // Help modal
        const helpBtn = document.getElementById('helpBtn');
        const helpModal = document.getElementById('helpModal');
        const closeModal = document.getElementById('closeModal');
        
        helpBtn.addEventListener('click', () => {
            helpModal.classList.add('active');
        });
        
        closeModal.addEventListener('click', () => {
            helpModal.classList.remove('active');
        });
        
        helpModal.addEventListener('click', (e) => {
            if (e.target === helpModal) {
                helpModal.classList.remove('active');
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                helpModal.classList.remove('active');
            }
            if (e.key === 'r' && !e.ctrlKey && !e.metaKey) {
                this.reset();
            }
        });
    }
    
    selectGroup(groupName) {
        this.currentGroup = groupName;
        this.currentCayleyNode = 'e';  // Reset to identity
        const group = WallpaperGroups[groupName];
        
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.group === groupName) {
                btn.classList.add('active');
            }
        });
        
        // Update header
        document.getElementById('groupName').textContent = groupName;
        document.getElementById('groupDescription').textContent = group.description;
        
        // Update properties
        const propsContainer = document.getElementById('groupProperties');
        propsContainer.innerHTML = `
            <span class="property-tag lattice">${group.lattice}</span>
            <span class="property-tag rotation">C${this.subscript(group.rotationOrder)}</span>
            ${group.hasReflection ? '<span class="property-tag" style="background: rgba(0,212,170,0.15); color: var(--crystal-emerald); border: 1px solid rgba(0,212,170,0.3);">σ</span>' : ''}
            ${group.hasGlide ? '<span class="property-tag" style="background: rgba(246,185,59,0.15); color: var(--crystal-amber); border: 1px solid rgba(246,185,59,0.3);">g</span>' : ''}
        `;
        
        // Update math panel
        this.updateMathPanel(group);
        
        // Reset operations
        this.reset();
        
        // Generate and display pattern with CORRECT symmetries
        this.generatePattern(groupName);
    }
    
    subscript(num) {
        const subscripts = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆'};
        return String(num).split('').map(d => subscripts[d] || d).join('');
    }
    
    updateMathPanel(group) {
        document.getElementById('mathGroupName').textContent = group.name;
        document.getElementById('mathLattice').textContent = group.lattice;
        
        const rotationAngles = {1: '360° (trivial)', 2: '180°', 3: '120°', 4: '90°', 6: '60°'};
        document.getElementById('mathRotation').textContent = 
            `${group.rotationOrder} (${rotationAngles[group.rotationOrder]})`;
        
        document.getElementById('mathReflection').textContent = group.hasReflection ? 'Sí' : 'No';
        document.getElementById('mathGlide').textContent = group.hasGlide ? 'Sí' : 'No';
        document.getElementById('mathPointGroupOrder').textContent = group.pointGroupOrder;
        document.getElementById('generatorsList').innerHTML = `<code>${group.generators}</code>`;
        
        // Update valid symmetries display
        this.updateValidSymmetries(group);
        
        // Update Cayley table
        this.updateCayleyTable(group);
        
        // Update Cayley graph
        this.drawCayleyGraph(group);
    }
    
    updateValidSymmetries(group) {
        const container = document.getElementById('validSymmetriesContainer');
        if (!container) return;
        
        let html = '<h4>Simetrías que dan 100%:</h4><ul class="valid-symmetries-list">';
        
        if (group.validSymmetries) {
            for (const sym of group.validSymmetries) {
                html += `<li><code>${sym.name}</code></li>`;
            }
        }
        
        html += '</ul>';
        
        if (group.invalidSymmetries && group.invalidSymmetries.length > 0) {
            html += '<h4>NO son simetrías:</h4><p class="invalid-symmetries">';
            html += group.invalidSymmetries.join(', ');
            html += '</p>';
        }
        
        container.innerHTML = html;
    }
    
    updateCayleyTable(group) {
        const container = document.getElementById('cayleyTableContainer');
        if (!container) return;
        
        if (!group.cayleyTable || typeof group.cayleyTable.table === 'string') {
            container.innerHTML = `<p class="cayley-note">${group.cayleyTable?.table || 'Tabla no disponible'}</p>`;
            return;
        }
        
        const elements = group.cayleyTable.elements;
        const table = group.cayleyTable.table;
        
        // Find identity element (usually 'e')
        const identitySymbols = ['e', 'E', 'I', 'Id', 'id'];
        
        let html = '<table class="cayley-table"><thead><tr><th>∘</th>';
        for (const el of elements) {
            html += `<th>${el}</th>`;
        }
        html += '</tr></thead><tbody>';
        
        for (let i = 0; i < elements.length; i++) {
            html += `<tr><th>${elements[i]}</th>`;
            for (let j = 0; j < elements.length; j++) {
                const result = table[i][j];
                // Check if result is identity (a·b = e means b is inverse of a)
                const isIdentity = identitySymbols.includes(result);
                // Check if it's a self-inverse (a·a = e) 
                const isSelfInverse = isIdentity && i === j;
                // Check if it's an inverse pair (a·b = e where a ≠ b)
                const isInversePair = isIdentity && i !== j;
                
                let cellClass = '';
                let tooltip = '';
                if (isSelfInverse) {
                    cellClass = 'cayley-self-inverse';
                    tooltip = `${elements[i]} es su propio inverso`;
                } else if (isInversePair) {
                    cellClass = 'cayley-inverse';
                    tooltip = `${elements[i]} × ${elements[j]} = e (son inversos)`;
                }
                
                html += `<td class="${cellClass}" ${tooltip ? `title="${tooltip}"` : ''} data-row="${elements[i]}" data-col="${elements[j]}" data-result="${result}">${result}</td>`;
            }
            html += '</tr>';
        }
        
        html += '</tbody></table>';
        
        // Add legend for highlighted cells
        html += `
            <div class="cayley-legend">
                <span class="legend-item"><span class="legend-box self-inverse"></span> A² = e (involutivo)</span>
                <span class="legend-item"><span class="legend-box inverse-pair"></span> A·B = e (inversos)</span>
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    /**
     * Draw Cayley graph for the group's point group
     * Nodes are group elements, edges show generator connections
     * Interactive: highlights current position after applying operations
     */
    drawCayleyGraph(group, highlightNode = 'e') {
        const canvas = document.getElementById('cayleyGraph');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear
        ctx.fillStyle = '#0f1419';
        ctx.fillRect(0, 0, width, height);
        
        if (!group.cayleyTable || typeof group.cayleyTable.table === 'string') {
            ctx.fillStyle = '#5c6874';
            ctx.font = '12px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText('Grafo no disponible para este grupo', width/2, height/2);
            return;
        }
        
        const elements = group.cayleyTable.elements;
        const n = elements.length;
        
        // Position nodes in a circle
        const centerX = width / 2;
        const centerY = height / 2 - 10;  // Shift up a bit for legend
        const radius = Math.min(width, height) / 2 - 60;
        
        this.cayleyNodes = [];
        for (let i = 0; i < n; i++) {
            const angle = (i * 2 * Math.PI / n) - Math.PI / 2;  // Start from top
            const label = elements[i];
            
            // Determine node type based on explicit symbols
            const isIdentity = label === 'e';
            const isRotation = label.startsWith('C') && !label.includes('σ');
            const isReflection = label.includes('σ');
            const isGlide = label.includes('g') && !label.includes('σ');
            const isTranslation = label.includes('t') || label.includes('T');
            
            this.cayleyNodes.push({
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle),
                label: label,
                isIdentity: isIdentity,
                isRotation: isRotation,
                isReflection: isReflection,
                isGlide: isGlide,
                isTranslation: isTranslation,
                isCurrent: label === highlightNode
            });
        }
        
        // Draw edges with arrows to show generator connections
        ctx.lineWidth = 1.5;
        
        // Draw rotation edges (connecting consecutive rotations)
        for (let i = 0; i < n; i++) {
            const j = (i + 1) % n;
            
            // Create curved edges for better visualization
            const gradient = ctx.createLinearGradient(
                this.cayleyNodes[i].x, this.cayleyNodes[i].y, 
                this.cayleyNodes[j].x, this.cayleyNodes[j].y
            );
            
            // Color by edge type
            if (this.cayleyNodes[i].isRotation || this.cayleyNodes[j].isRotation) {
                gradient.addColorStop(0, 'rgba(183, 148, 246, 0.7)');
                gradient.addColorStop(1, 'rgba(183, 148, 246, 0.4)');
            } else if (this.cayleyNodes[i].isReflection || this.cayleyNodes[j].isReflection) {
                gradient.addColorStop(0, 'rgba(79, 168, 199, 0.7)');
                gradient.addColorStop(1, 'rgba(79, 168, 199, 0.4)');
            } else {
                gradient.addColorStop(0, 'rgba(0, 212, 170, 0.7)');
                gradient.addColorStop(1, 'rgba(0, 212, 170, 0.4)');
            }
            
            ctx.strokeStyle = gradient;
            ctx.beginPath();
            ctx.moveTo(this.cayleyNodes[i].x, this.cayleyNodes[i].y);
            ctx.lineTo(this.cayleyNodes[j].x, this.cayleyNodes[j].y);
            ctx.stroke();
            
            // Draw arrow head
            this.drawArrowHead(ctx, 
                this.cayleyNodes[i].x, this.cayleyNodes[i].y,
                this.cayleyNodes[j].x, this.cayleyNodes[j].y,
                gradient
            );
        }
        
        // Draw reflection edges (involutions connect back to identity)
        for (let i = 0; i < n; i++) {
            if (this.cayleyNodes[i].isReflection && !this.cayleyNodes[i].isIdentity) {
                ctx.strokeStyle = 'rgba(79, 168, 199, 0.35)';
                ctx.setLineDash([4, 4]);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(this.cayleyNodes[i].x, this.cayleyNodes[i].y);
                ctx.lineTo(this.cayleyNodes[0].x, this.cayleyNodes[0].y);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.lineWidth = 1.5;
            }
        }
        
        // Draw nodes
        const nodeRadius = 18;
        
        for (const node of this.cayleyNodes) {
            // Glow effect (stronger for current node)
            const glowRadius = node.isCurrent ? nodeRadius * 2.5 : nodeRadius * 1.5;
            const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, glowRadius);
            
            // Color scheme:
            // - Identity: Green (#00d4aa)
            // - Rotation: Purple (#b794f6)
            // - Reflection: Blue (#4fa8c7)
            // - Glide: Teal (#20c997)
            // - Translation: Gray (#8b98a5)
            // - Current: Amber (#f6b93b)
            
            if (node.isCurrent) {
                glow.addColorStop(0, 'rgba(246, 185, 59, 0.8)');
                glow.addColorStop(0.5, 'rgba(246, 185, 59, 0.3)');
                glow.addColorStop(1, 'rgba(246, 185, 59, 0)');
            } else if (node.isIdentity) {
                glow.addColorStop(0, 'rgba(0, 212, 170, 0.4)');
                glow.addColorStop(1, 'rgba(0, 212, 170, 0)');
            } else if (node.isRotation) {
                glow.addColorStop(0, 'rgba(183, 148, 246, 0.4)');
                glow.addColorStop(1, 'rgba(183, 148, 246, 0)');
            } else if (node.isGlide) {
                glow.addColorStop(0, 'rgba(32, 201, 151, 0.4)');
                glow.addColorStop(1, 'rgba(32, 201, 151, 0)');
            } else if (node.isTranslation) {
                glow.addColorStop(0, 'rgba(139, 152, 165, 0.4)');
                glow.addColorStop(1, 'rgba(139, 152, 165, 0)');
            } else {
                glow.addColorStop(0, 'rgba(79, 168, 199, 0.4)');
                glow.addColorStop(1, 'rgba(79, 168, 199, 0)');
            }
            
            ctx.fillStyle = glow;
            ctx.beginPath();
            ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
            ctx.fill();
            
            // Node circle
            let nodeColor;
            if (node.isCurrent) {
                nodeColor = '#f6b93b';  // Amber for current
            } else if (node.isIdentity) {
                nodeColor = '#00d4aa';  // Green
            } else if (node.isRotation) {
                nodeColor = '#b794f6';  // Purple
            } else if (node.isGlide) {
                nodeColor = '#20c997';  // Teal
            } else if (node.isTranslation) {
                nodeColor = '#8b98a5';  // Gray
            } else {
                nodeColor = '#4fa8c7';  // Blue (reflection)
            }
            
            ctx.fillStyle = nodeColor;
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.isCurrent ? nodeRadius + 3 : nodeRadius, 0, Math.PI * 2);
            ctx.fill();
            
            // Border
            ctx.strokeStyle = node.isCurrent ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = node.isCurrent ? 3 : 2;
            ctx.stroke();
            
            // Label
            ctx.fillStyle = '#0f1419';
            ctx.font = node.isCurrent ? 'bold 12px JetBrains Mono' : 'bold 10px JetBrains Mono';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(node.label, node.x, node.y);
        }
        
        // Info at bottom
        ctx.fillStyle = '#8b98a5';
        ctx.font = '9px Outfit';
        ctx.textAlign = 'center';
        ctx.fillText(`Grupo Puntual: orden ${n}`, width/2, height - 20);
        
        // Legend explanation
        ctx.fillStyle = '#5c6874';
        ctx.font = '8px Outfit';
        ctx.fillText('→ Rotación   ⤏ Reflexión (σ²=e)', width/2, height - 8);
    }
    
    /**
     * Draw arrow head for Cayley graph edges
     */
    drawArrowHead(ctx, fromX, fromY, toX, toY, color) {
        const headLen = 8;
        const dx = toX - fromX;
        const dy = toY - fromY;
        const angle = Math.atan2(dy, dx);
        
        // Position arrow at 70% of the edge (not at the node)
        const arrowX = fromX + dx * 0.6;
        const arrowY = fromY + dy * 0.6;
        
        ctx.fillStyle = typeof color === 'string' ? color : 'rgba(183, 148, 246, 0.7)';
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(arrowX - headLen * Math.cos(angle - Math.PI / 6), arrowY - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(arrowX - headLen * Math.cos(angle + Math.PI / 6), arrowY - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    }
    
    /**
     * Update Cayley graph to highlight current position after operation
     */
    updateCayleyGraphPosition(operationName) {
        const group = WallpaperGroups[this.currentGroup];
        if (!group.cayleyTable || typeof group.cayleyTable.table === 'string') return;
        
        const elements = group.cayleyTable.elements;
        const table = group.cayleyTable.table;
        
        // Parse the operation name more robustly
        // Format: "Rotación 120°" or "Reflexión Vertical" etc.
        const opName = operationName.trim();
        
        // Map operation names to possible Cayley table element names
        // Uses specific symbols like σᵥ, σₕ, gₕ, gᵥ for clarity
        const opMapping = {
            'Rotación 60°': ['C₆'],
            'Rotación 90°': ['C₄'],
            'Rotación 120°': ['C₃'],
            'Rotación 180°': ['C₂'],
            'Rotación 240°': ['C₃²'],
            'Rotación 270°': ['C₄³'],
            'Rotación 300°': ['C₆⁵'],
            'Reflexión Vertical': ['σᵥ', 'σᵥ(0°)', 'σ'],
            'Reflexión Horizontal': ['σₕ', 'σₕ(0°)', 'σ'],
            'Reflexión Diagonal': ['σ_d', 'σ_d\''],
            'Reflexión Anti-diagonal': ['σ_d\'', 'σ_d'],
            'Glide Horizontal': ['gₕ', 'g'],
            'Glide Vertical': ['gᵥ', 'g']
        };
        
        const possibleElements = opMapping[opName];
        if (!possibleElements) {
            console.log(`[Cayley] No mapping for operation: "${opName}"`);
            return;
        }
        
        // Find which element from the mapping exists in this group's table
        let opIdx = -1;
        let matchedElement = null;
        for (const elem of possibleElements) {
            opIdx = elements.indexOf(elem);
            if (opIdx !== -1) {
                matchedElement = elem;
                break;
            }
        }
        
        if (opIdx === -1) {
            console.log(`[Cayley] No element found for "${opName}" in elements:`, elements);
            return;
        }
        
        // Find current position in elements array
        const currentIdx = elements.indexOf(this.currentCayleyNode);
        
        if (currentIdx === -1) {
            console.log(`[Cayley] Current node "${this.currentCayleyNode}" not found in elements`);
            return;
        }
        
        // Apply the operation: new position = current × operation
        const previousNode = this.currentCayleyNode;
        this.currentCayleyNode = table[currentIdx][opIdx];
        
        console.log(`[Cayley] Transition: ${previousNode} × ${matchedElement} = ${this.currentCayleyNode}`);
        
        // Redraw graph with new highlight
        this.drawCayleyGraph(group, this.currentCayleyNode);
    }
    
    updateMatrixDisplay() {
        const formatted = SymmetryOps.formatMatrix(this.currentMatrix);
        document.getElementById('m00').textContent = formatted.m00;
        document.getElementById('m01').textContent = formatted.m01;
        document.getElementById('m10').textContent = formatted.m10;
        document.getElementById('m11').textContent = formatted.m11;
    }
    
    /**
     * Generate a mathematically correct pattern for the given wallpaper group.
     * 
     * CRITICAL: The pattern must have EXACTLY the symmetries of the group.
     * All patterns are generated CENTERED on the canvas center for exact symmetry.
     * - p1: NO rotational symmetry, NO reflection
     * - p2: C2 symmetry (180°) ONLY
     * - etc.
     */
    generatePattern(groupName) {
        const size = this.canvasSize;
        const imageData = this.originalCtx.createImageData(size, size);
        const group = WallpaperGroups[groupName];
        
        // Use different seeds for different groups to ensure variety
        const seed = this.hashString(groupName + '_v3_centered');
        const rng = this.seededRandom(seed);
        
        // Generate pattern with EXACT symmetries centered on canvas
        let pattern;
        
        switch (groupName) {
            case 'p1':
                pattern = this.generateP1(size, rng);
                break;
            case 'p2':
                pattern = this.generateP2Centered(size, rng);
                break;
            case 'pm':
                pattern = this.generatePMCentered(size, rng);
                break;
            case 'pg':
                pattern = this.generatePG(size, rng);
                break;
            case 'cm':
                pattern = this.generateCM(size, rng);
                break;
            case 'pmm':
                pattern = this.generatePMMCentered(size, rng);
                break;
            case 'pmg':
                pattern = this.generatePMG(size, rng);
                break;
            case 'pgg':
                pattern = this.generatePGGCentered(size, rng);
                break;
            case 'cmm':
                pattern = this.generateCMMCentered(size, rng);
                break;
            case 'p4':
                pattern = this.generateP4Centered(size, rng);
                break;
            case 'p4m':
                pattern = this.generateP4MCentered(size, rng);
                break;
            case 'p4g':
                pattern = this.generateP4GCentered(size, rng);
                break;
            case 'p3':
                pattern = this.generateP3Centered(size, rng);
                break;
            case 'p3m1':
                pattern = this.generateP3M1Centered(size, rng);
                break;
            case 'p31m':
                pattern = this.generateP31MCentered(size, rng);
                break;
            case 'p6':
                pattern = this.generateP6Centered(size, rng);
                break;
            case 'p6m':
                pattern = this.generateP6MCentered(size, rng);
                break;
            default:
                pattern = this.generateP1(size, rng);
        }
        
        // Fill image data with viridis colormap
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const value = pattern[y * size + x];
                const [r, g, b] = this.viridisColor(value);
                const idx = (y * size + x) * 4;
                imageData.data[idx] = r;
                imageData.data[idx + 1] = g;
                imageData.data[idx + 2] = b;
                imageData.data[idx + 3] = 255;
            }
        }
        
        // Store and display
        this.originalImageData = imageData.data.slice();
        this.transformedImageData = imageData.data.slice();
        
        this.originalCtx.putImageData(imageData, 0, 0);
        this.transformedCtx.putImageData(imageData, 0, 0);
        this.updateDifference();
    }
    
    /**
     * Create a truly ASYMMETRIC motif
     * This is crucial - the motif must have NO symmetry
     */
    createAsymmetricMotif(size, rng) {
        const motif = new Float32Array(size * size);
        
        // Use asymmetric shapes: different sized blobs at non-symmetric positions
        const numElements = 4 + Math.floor(rng() * 3);
        
        for (let i = 0; i < numElements; i++) {
            // Avoid symmetric positions
            const cx = rng() * size * 0.7 + size * 0.1;
            const cy = rng() * size * 0.7 + size * 0.1;
            
            // Elliptical Gaussian with random orientation (asymmetric)
            const sigmaX = 5 + rng() * 15;
            const sigmaY = 5 + rng() * 15;  // Different from sigmaX
            const angle = rng() * Math.PI;  // Random rotation
            const amplitude = 0.3 + rng() * 0.7;
            
            // Add an asymmetric tail to break any remaining symmetry
            const tailDx = (rng() - 0.5) * size * 0.3;
            const tailDy = (rng() - 0.5) * size * 0.3;
            
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const dx = x - cx;
                    const dy = y - cy;
                    
                    // Rotated elliptical Gaussian
                    const rx = dx * Math.cos(angle) + dy * Math.sin(angle);
                    const ry = -dx * Math.sin(angle) + dy * Math.cos(angle);
                    
                    let value = amplitude * Math.exp(-(rx*rx)/(2*sigmaX*sigmaX) - (ry*ry)/(2*sigmaY*sigmaY));
                    
                    // Add asymmetric tail
                    const tdx = x - (cx + tailDx);
                    const tdy = y - (cy + tailDy);
                    value += amplitude * 0.3 * Math.exp(-(tdx*tdx + tdy*tdy)/(2*sigmaX*sigmaX));
                    
                    motif[y * size + x] += value;
                }
            }
        }
        
        return motif;
    }
    
    /**
     * p1: Only translations - NO rotational or reflection symmetry
     */
    generateP1(size, rng) {
        const cellSize = Math.floor(size / 3);
        const motif = this.createAsymmetricMotif(cellSize, rng);
        const pattern = new Float32Array(size * size);
        
        // Tile the asymmetric motif - NO symmetry operations applied
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const mx = x % cellSize;
                const my = y % cellSize;
                pattern[y * size + x] = motif[my * cellSize + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p2: 180° rotation symmetry ONLY (no reflections)
     */
    generateP2(size, rng) {
        const cellSize = Math.floor(size / 3);
        const halfCell = Math.floor(cellSize / 2);
        
        // Create asymmetric motif for half the cell
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Determine if we're in the "original" or "rotated" half
                // Split diagonally to avoid creating reflection symmetry
                const inFirstHalf = (cx + cy) < cellSize;
                
                let mx, my;
                if (inFirstHalf) {
                    mx = cx % halfCell;
                    my = cy % halfCell;
                } else {
                    // 180° rotation: (x,y) -> (cellSize-1-x, cellSize-1-y)
                    mx = (cellSize - 1 - cx) % halfCell;
                    my = (cellSize - 1 - cy) % halfCell;
                }
                
                mx = Math.max(0, Math.min(halfCell - 1, mx));
                my = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[my * halfCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pm: Vertical reflection symmetry ONLY (no rotation)
     */
    generatePM(size, rng) {
        const cellSize = Math.floor(size / 3);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Reflect across vertical center of cell
                const mx = cx < halfCell ? cx : (cellSize - 1 - cx);
                const my = cy % halfCell;
                
                const mxClamped = Math.max(0, Math.min(halfCell - 1, mx));
                const myClamped = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[myClamped * halfCell + mxClamped];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pg: Glide reflection (NOT pure reflection)
     */
    generatePG(size, rng) {
        const cellSize = Math.floor(size / 3);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                let mx, my;
                if (cx < halfCell) {
                    // Original
                    mx = cx;
                    my = cy % halfCell;
                } else {
                    // Glide: reflect vertically + shift by half cell in y
                    mx = cellSize - 1 - cx;
                    my = (cy + halfCell) % cellSize;
                    my = my % halfCell;
                }
                
                mx = Math.max(0, Math.min(halfCell - 1, mx));
                my = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[my * halfCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * cm: Reflection + centered cell
     */
    generateCM(size, rng) {
        const cellSize = Math.floor(size / 3);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Centered cell: shift by half in alternate rows
                const rowShift = (Math.floor(y / cellSize) % 2) * halfCell;
                const effectiveCx = (cx + rowShift) % cellSize;
                
                // Reflect
                const mx = effectiveCx < halfCell ? effectiveCx : (cellSize - 1 - effectiveCx);
                const my = cy % halfCell;
                
                const mxClamped = Math.max(0, Math.min(halfCell - 1, mx));
                const myClamped = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[myClamped * halfCell + mxClamped];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pmm: Two perpendicular reflections (implies C2)
     */
    generatePMM(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Reflect in both x and y
                const mx = cx < halfCell ? cx : (cellSize - 1 - cx);
                const my = cy < halfCell ? cy : (cellSize - 1 - cy);
                
                const mxClamped = Math.max(0, Math.min(halfCell - 1, mx));
                const myClamped = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[myClamped * halfCell + mxClamped];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pmg: Reflection + glide perpendicular
     */
    generatePMG(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Vertical reflection
                let mx = cx < halfCell ? cx : (cellSize - 1 - cx);
                
                // Glide in horizontal direction based on which half
                let my;
                if (cy < halfCell) {
                    my = cy;
                } else {
                    // Shift by half when in glide region
                    my = cellSize - 1 - cy;
                    if (cx >= halfCell) {
                        my = (my + halfCell) % cellSize;
                    }
                }
                my = my % halfCell;
                
                mx = Math.max(0, Math.min(halfCell - 1, mx));
                my = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[my * halfCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pgg: Two perpendicular glides (implies C2, no pure reflection)
     */
    generatePGG(size, rng) {
        const cellSize = Math.floor(size / 4);
        const halfCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(halfCell, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                const qx = cx >= halfCell ? 1 : 0;
                const qy = cy >= halfCell ? 1 : 0;
                
                let mx = cx % halfCell;
                let my = cy % halfCell;
                
                // Glide operations create 180° rotation at intersections
                if ((qx + qy) % 2 === 1) {
                    mx = halfCell - 1 - mx;
                    my = halfCell - 1 - my;
                }
                
                mx = Math.max(0, Math.min(halfCell - 1, mx));
                my = Math.max(0, Math.min(halfCell - 1, my));
                
                pattern[y * size + x] = motif[my * halfCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * cmm: Centered with perpendicular reflections
     */
    generateCMM(size, rng) {
        // cmm is like pmm but with centered cell
        return this.generatePMM(size, rng);
    }
    
    /**
     * p4: 90° rotation symmetry (no reflections)
     */
    generateP4(size, rng) {
        const cellSize = Math.floor(size / 3);
        const quarterCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(quarterCell, rng);
        const pattern = new Float32Array(size * size);
        
        const center = cellSize / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // Determine quadrant and apply rotation
                const dx = cx - center;
                const dy = cy - center;
                
                let mx, my;
                
                if (dx >= 0 && dy < 0) {
                    // Quadrant 1 (top-right): original
                    mx = cx % quarterCell;
                    my = cy % quarterCell;
                } else if (dx >= 0 && dy >= 0) {
                    // Quadrant 2 (bottom-right): 90° rotation
                    mx = (cellSize - 1 - cy) % quarterCell;
                    my = cx % quarterCell;
                } else if (dx < 0 && dy >= 0) {
                    // Quadrant 3 (bottom-left): 180° rotation
                    mx = (cellSize - 1 - cx) % quarterCell;
                    my = (cellSize - 1 - cy) % quarterCell;
                } else {
                    // Quadrant 4 (top-left): 270° rotation
                    mx = cy % quarterCell;
                    my = (cellSize - 1 - cx) % quarterCell;
                }
                
                mx = Math.max(0, Math.min(quarterCell - 1, mx));
                my = Math.max(0, Math.min(quarterCell - 1, my));
                
                pattern[y * size + x] = motif[my * quarterCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4m: 90° rotation + 4 reflection axes
     */
    generateP4M(size, rng) {
        const cellSize = Math.floor(size / 3);
        const eighthCell = Math.floor(cellSize / 4);
        const motif = this.createAsymmetricMotif(eighthCell, rng);
        const pattern = new Float32Array(size * size);
        
        const center = cellSize / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                // First apply reflections to get to first octant
                let rx = cx < center ? cx : (cellSize - 1 - cx);
                let ry = cy < center ? cy : (cellSize - 1 - cy);
                
                // Diagonal reflection if needed
                if (rx > ry) {
                    [rx, ry] = [ry, rx];
                }
                
                const mx = Math.max(0, Math.min(eighthCell - 1, rx % eighthCell));
                const my = Math.max(0, Math.min(eighthCell - 1, ry % eighthCell));
                
                pattern[y * size + x] = motif[my * eighthCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4g: 90° rotation + diagonal reflections + glides
     */
    generateP4G(size, rng) {
        const cellSize = Math.floor(size / 3);
        const quarterCell = Math.floor(cellSize / 2);
        const motif = this.createAsymmetricMotif(quarterCell, rng);
        const pattern = new Float32Array(size * size);
        
        const center = cellSize / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const cx = x % cellSize;
                const cy = y % cellSize;
                
                const dx = cx - center;
                const dy = cy - center;
                
                let mx, my;
                
                // Determine quadrant
                const qx = dx >= 0 ? 1 : 0;
                const qy = dy >= 0 ? 1 : 0;
                const quadrant = qy * 2 + qx;
                
                mx = Math.abs(dx) % quarterCell;
                my = Math.abs(dy) % quarterCell;
                
                // Apply rotation based on quadrant
                switch (quadrant) {
                    case 1: // 90°
                        [mx, my] = [quarterCell - 1 - my, mx];
                        break;
                    case 3: // 180°
                        mx = quarterCell - 1 - mx;
                        my = quarterCell - 1 - my;
                        break;
                    case 2: // 270°
                        [mx, my] = [my, quarterCell - 1 - mx];
                        break;
                }
                
                // Diagonal reflection for p4g
                if ((qx + qy) % 2 === 1) {
                    [mx, my] = [my, mx];
                }
                
                mx = Math.max(0, Math.min(quarterCell - 1, mx));
                my = Math.max(0, Math.min(quarterCell - 1, my));
                
                pattern[y * size + x] = motif[my * quarterCell + mx];
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p3: 120° rotation (no reflections)
     */
    generateP3(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Create asymmetric elements in first 120° sector
        const elements = [];
        const numElements = 3 + Math.floor(rng() * 2);
        
        for (let i = 0; i < numElements; i++) {
            const r = 20 + rng() * (size / 4);
            const theta = rng() * (2 * Math.PI / 3);  // First third only
            elements.push({
                x: cx + r * Math.cos(theta),
                y: cy + r * Math.sin(theta),
                sigmaX: 10 + rng() * 15,
                sigmaY: 8 + rng() * 12,
                angle: rng() * Math.PI,
                amplitude: 0.3 + rng() * 0.7
            });
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = 0;
                
                // Apply 3-fold rotation
                for (let rot = 0; rot < 3; rot++) {
                    const rotAngle = rot * 2 * Math.PI / 3;
                    const cos = Math.cos(rotAngle);
                    const sin = Math.sin(rotAngle);
                    
                    for (const el of elements) {
                        const ex = cx + (el.x - cx) * cos - (el.y - cy) * sin;
                        const ey = cy + (el.x - cx) * sin + (el.y - cy) * cos;
                        
                        const dx = x - ex;
                        const dy = y - ey;
                        const rx = dx * Math.cos(el.angle) + dy * Math.sin(el.angle);
                        const ry = -dx * Math.sin(el.angle) + dy * Math.cos(el.angle);
                        
                        value += el.amplitude * Math.exp(-(rx*rx)/(2*el.sigmaX*el.sigmaX) - (ry*ry)/(2*el.sigmaY*el.sigmaY));
                    }
                }
                
                pattern[y * size + x] = value / 3;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p3m1: 120° rotation + reflections through centers
     */
    generateP3M1(size, rng) {
        const base = this.generateP3(size, rng);
        const pattern = new Float32Array(size * size);
        
        const cx = size / 2;
        const cy = size / 2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Original + vertical reflection through center
                const rx = size - 1 - x;
                pattern[y * size + x] = (base[y * size + x] + base[y * size + rx]) / 2;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p31m: 120° rotation + reflections between centers
     */
    generateP31M(size, rng) {
        const base = this.generateP3(size, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Original + horizontal reflection
                const ry = size - 1 - y;
                pattern[y * size + x] = (base[y * size + x] + base[ry * size + x]) / 2;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p6: 60° rotation (no reflections)
     */
    generateP6(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Create elements in first 60° sector
        const elements = [];
        const numElements = 2 + Math.floor(rng() * 2);
        
        for (let i = 0; i < numElements; i++) {
            const r = 20 + rng() * (size / 4);
            const theta = rng() * (Math.PI / 3);  // First sixth only
            elements.push({
                x: cx + r * Math.cos(theta),
                y: cy + r * Math.sin(theta),
                sigmaX: 8 + rng() * 12,
                sigmaY: 6 + rng() * 10,
                angle: rng() * Math.PI,
                amplitude: 0.3 + rng() * 0.7
            });
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = 0;
                
                // Apply 6-fold rotation
                for (let rot = 0; rot < 6; rot++) {
                    const rotAngle = rot * Math.PI / 3;
                    const cos = Math.cos(rotAngle);
                    const sin = Math.sin(rotAngle);
                    
                    for (const el of elements) {
                        const ex = cx + (el.x - cx) * cos - (el.y - cy) * sin;
                        const ey = cy + (el.x - cx) * sin + (el.y - cy) * cos;
                        
                        const dx = x - ex;
                        const dy = y - ey;
                        const rx = dx * Math.cos(el.angle) + dy * Math.sin(el.angle);
                        const ry = -dx * Math.sin(el.angle) + dy * Math.cos(el.angle);
                        
                        value += el.amplitude * Math.exp(-(rx*rx)/(2*el.sigmaX*el.sigmaX) - (ry*ry)/(2*el.sigmaY*el.sigmaY));
                    }
                }
                
                pattern[y * size + x] = value / 6;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p6m: 60° rotation + 6 reflection axes (maximum symmetry)
     */
    generateP6M(size, rng) {
        const base = this.generateP6(size, rng);
        const pattern = new Float32Array(size * size);
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                // Apply both vertical and horizontal reflections
                const rx = size - 1 - x;
                const ry = size - 1 - y;
                
                const v1 = base[y * size + x];
                const v2 = base[y * size + rx];
                const v3 = base[ry * size + x];
                const v4 = base[ry * size + rx];
                
                pattern[y * size + x] = (v1 + v2 + v3 + v4) / 4;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    normalizePattern(pattern) {
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < pattern.length; i++) {
            if (pattern[i] < min) min = pattern[i];
            if (pattern[i] > max) max = pattern[i];
        }
        
        const range = max - min || 1;
        for (let i = 0; i < pattern.length; i++) {
            pattern[i] = (pattern[i] - min) / range;
        }
        
        return pattern;
    }
    
    /**
     * p2: 180° rotation centered - pattern symmetric under 180° rotation around canvas center
     */
    generateP2Centered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
        // Create random asymmetric elements in one half
        const elements = [];
        const numElements = 4 + Math.floor(rng() * 3);
        
        for (let i = 0; i < numElements; i++) {
            elements.push({
                x: rng() * size * 0.4,  // Left half
                y: rng() * size,
                sigma: 15 + rng() * 25,
                amplitude: 0.3 + rng() * 0.7
            });
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = 0;
                
                // Original elements
                for (const el of elements) {
                    const dx = x - el.x;
                    const dy = y - el.y;
                    value += el.amplitude * Math.exp(-(dx*dx + dy*dy)/(2*el.sigma*el.sigma));
                }
                
                // 180° rotated elements (around center)
                for (const el of elements) {
                    const rx = 2*cx - el.x;
                    const ry = 2*cy - el.y;
                    const dx = x - rx;
                    const dy = y - ry;
                    value += el.amplitude * Math.exp(-(dx*dx + dy*dy)/(2*el.sigma*el.sigma));
                }
                
                pattern[y * size + x] = value;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pm: Vertical reflection centered - pattern symmetric under vertical reflection
     */
    generatePMCentered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = (size - 1) / 2;
        
        const elements = [];
        const numElements = 5 + Math.floor(rng() * 3);
        
        for (let i = 0; i < numElements; i++) {
            elements.push({
                x: rng() * size * 0.5,  // Left half only
                y: rng() * size,
                sigma: 15 + rng() * 25,
                amplitude: 0.3 + rng() * 0.7
            });
        }
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                let value = 0;
                
                for (const el of elements) {
                    // Original
                    const dx1 = x - el.x;
                    const dy1 = y - el.y;
                    value += el.amplitude * Math.exp(-(dx1*dx1 + dy1*dy1)/(2*el.sigma*el.sigma));
                    
                    // Reflected across vertical center
                    const dx2 = x - (2*cx - el.x);
                    const dy2 = y - el.y;
                    value += el.amplitude * Math.exp(-(dx2*dx2 + dy2*dy2)/(2*el.sigma*el.sigma));
                }
                
                pattern[y * size + x] = value;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pmm: Both reflections centered
     */
    generatePMMCentered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Use cos(2nθ) for exact 2-fold symmetry + reflections
        const n1 = 1 + Math.floor(rng() * 2);
        const amp1 = 0.2 + rng() * 0.25;
        const amp2 = 0.1 + rng() * 0.15;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(2nθ) has 2-fold rotation + 2 reflection axes
                const radial = Math.exp(-r * r / (2 * 55 * 55));
                const angular = 0.5 
                    + amp1 * Math.cos(2 * theta)
                    + amp2 * Math.cos(4 * theta)
                    + 0.1 * Math.cos(2 * n1 * theta);
                
                const radialVar = 1 + 0.2 * Math.cos(r / 25);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * pgg: Two perpendicular glides centered - uses cos(2nθ) for exact 180° symmetry
     */
    generatePGGCentered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Use cos(2nθ) for exact 2-fold symmetry
        const n1 = 1 + Math.floor(rng() * 2);
        const amp1 = 0.2 + rng() * 0.2;
        const amp2 = 0.1 + rng() * 0.15;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(2nθ) has exact 2-fold rotation symmetry
                const radial = Math.exp(-r * r / (2 * 55 * 55));
                const angular = 0.5 
                    + amp1 * Math.cos(2 * theta)
                    + amp2 * Math.cos(4 * theta)
                    + 0.08 * Math.cos(2 * n1 * theta);
                
                const radialVar = 1 + 0.2 * Math.cos(r / 22);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * cmm: Centered cell with reflections
     */
    generateCMMCentered(size, rng) {
        return this.generatePMMCentered(size, rng);
    }
    
    /**
     * p4: 90° rotation centered - EXACT 4-fold symmetry
     */
    generateP4Centered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Use cos(4nθ) for exact 4-fold rotational symmetry
        const n1 = 1 + Math.floor(rng() * 2);  // 1-2, so 4n = 4 or 8
        const n2 = 2 + Math.floor(rng() * 2);  // 2-3, so 4n = 8 or 12
        const amp1 = 0.2 + rng() * 0.25;
        const amp2 = 0.1 + rng() * 0.15;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(4nθ) has exact 4-fold symmetry: f(θ) = f(θ + 90°)
                // NO phase offsets to preserve exact symmetry
                const radial = Math.exp(-r * r / (2 * 55 * 55));
                const angular = 0.5 
                    + amp1 * Math.cos(4 * theta)
                    + amp2 * Math.cos(4 * n1 * theta)
                    + 0.1 * Math.cos(4 * n2 * theta);
                
                const radialVar = 1 + 0.2 * Math.cos(n1 * r / 25);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4m: 90° rotation + 4 reflections centered
     * Uses cos(4nθ) for exact 4-fold and reflection symmetry
     */
    generateP4MCentered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        const n1 = 1 + Math.floor(rng() * 2);
        const amp1 = 0.2 + rng() * 0.2;
        const amp2 = 0.1 + rng() * 0.15;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(4nθ) has 4-fold rotation + 4 reflection axes (D4)
                const radial = Math.exp(-r * r / (2 * 55 * 55));
                const angular = 0.5 
                    + amp1 * Math.cos(4 * theta)
                    + amp2 * Math.cos(8 * theta)
                    + 0.1 * Math.cos(4 * n1 * theta);
                
                const radialVar = 1 + 0.2 * Math.cos(r / 22);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p4g: 90° rotation + diagonal reflections centered
     * Uses cos(4nθ) for exact 4-fold and diagonal reflection symmetry
     */
    generateP4GCentered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        const n1 = 1 + Math.floor(rng() * 2);
        const amp1 = 0.15 + rng() * 0.2;
        const amp2 = 0.1 + rng() * 0.15;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(4nθ) has 4-fold rotation + diagonal reflections
                const radial = Math.exp(-r * r / (2 * 55 * 55));
                const angular = 0.5 
                    + amp1 * Math.cos(4 * theta)
                    + amp2 * Math.cos(8 * theta)
                    + 0.08 * Math.cos(4 * n1 * theta);
                
                const radialVar = 1 + 0.2 * Math.cos(r / 20);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p3: 120° rotation centered - EXACT 3-fold symmetry
     * Uses polar coordinates with cos(3θ) to ensure mathematical exactness
     */
    generateP3Centered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Use only multiples of 3 for angular frequency to preserve 3-fold symmetry
        // cos(3n*θ) is invariant under 120° rotation for any integer n
        const n1 = 1 + Math.floor(rng() * 2);  // 1-2, so 3n = 3 or 6
        const n2 = 1 + Math.floor(rng() * 2);
        const radialFreq = 1 + Math.floor(rng() * 2);
        
        // Amplitudes for variety (these don't affect symmetry)
        const amp1 = 0.2 + rng() * 0.3;
        const amp2 = 0.1 + rng() * 0.2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // Use ONLY cos(3n*θ) terms - these have EXACT 3-fold symmetry
                // NO phase offsets that could break symmetry
                const radial = Math.exp(-r * r / (2 * 60 * 60));
                const angular = 0.5 
                    + amp1 * Math.cos(3 * n1 * theta)
                    + amp2 * Math.cos(3 * n2 * theta);
                
                // Radial variation (doesn't affect rotational symmetry)
                const radialVar = 1 + 0.3 * Math.cos(radialFreq * r / 20);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p3m1: 120° rotation + 3 reflections through rotation centers
     * Uses cos(3nθ) terms with NO phase offsets for exact symmetry
     */
    generateP3M1Centered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        const n1 = 1 + Math.floor(rng() * 2);
        const amp1 = 0.2 + rng() * 0.3;
        const amp2 = 0.1 + rng() * 0.2;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(3nθ) has exact 3-fold rotation symmetry
                // Reflections are at θ = 0, 60°, 120° (through rotation centers)
                const radial = Math.exp(-r * r / (2 * 55 * 55));
                const angular = 0.5 
                    + amp1 * Math.cos(3 * theta)
                    + amp2 * Math.cos(6 * theta)
                    + 0.1 * Math.cos(3 * n1 * theta);
                
                const radialVar = 1 + 0.2 * Math.cos(r / 22);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p31m: 120° rotation + 3 reflections between rotation centers
     * Uses cos(3nθ) terms - same math as p3m1 but different visual
     */
    generateP31MCentered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        const n1 = 1 + Math.floor(rng() * 2);
        const amp1 = 0.15 + rng() * 0.25;
        const amp2 = 0.1 + rng() * 0.15;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(3nθ) has exact 3-fold rotation symmetry
                // Use different coefficients than p3m1 for visual variety
                const radial = Math.exp(-r * r / (2 * 55 * 55));
                const angular = 0.5 
                    + amp1 * Math.cos(3 * theta)
                    + amp2 * Math.cos(9 * theta)
                    + 0.1 * Math.cos(3 * n1 * theta);
                
                const radialVar = 1 + 0.25 * Math.cos(r / 18);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p6: 60° rotation centered - EXACT 6-fold symmetry
     * Uses cos(6nθ) for mathematical exactness - NO phase offsets
     */
    generateP6Centered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        // Only multiples of 6 for angular frequency to preserve 6-fold symmetry
        const n1 = 1 + Math.floor(rng() * 2);  // 1-2, so 6n = 6 or 12
        const n2 = 2 + Math.floor(rng() * 2);  // 2-3, so 6n = 12 or 18
        const amp1 = 0.2 + rng() * 0.2;
        const amp2 = 0.1 + rng() * 0.15;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(6nθ) has exact 6-fold symmetry: f(θ) = f(θ + 60°)
                // NO phase offsets to preserve exact symmetry
                const radial = Math.exp(-r * r / (2 * 55 * 55));
                const angular = 0.5 
                    + amp1 * Math.cos(6 * theta)
                    + amp2 * Math.cos(6 * n1 * theta)
                    + 0.1 * Math.cos(6 * n2 * theta);
                
                const radialVar = 1 + 0.2 * Math.cos(n1 * r / 25);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    /**
     * p6m: 60° rotation + 6 reflections (maximum symmetry)
     * Uses cos(6nθ) terms - NO phase offsets for exact symmetry
     */
    generateP6MCentered(size, rng) {
        const pattern = new Float32Array(size * size);
        const cx = size / 2;
        const cy = size / 2;
        
        const n1 = 1 + Math.floor(rng() * 2);
        const amp1 = 0.2 + rng() * 0.2;
        const amp2 = 0.1 + rng() * 0.15;
        
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const r = Math.sqrt(dx * dx + dy * dy);
                const theta = Math.atan2(dy, dx);
                
                // cos(6nθ) has 6-fold rotational symmetry
                // Also has reflection symmetry at θ = 0, 30°, 60°, etc.
                const radial = Math.exp(-r * r / (2 * 50 * 50));
                const angular = 0.5 
                    + amp1 * Math.cos(6 * theta)
                    + amp2 * Math.cos(12 * theta)
                    + 0.1 * Math.cos(6 * n1 * theta);
                
                const radialVar = 1 + 0.25 * Math.cos(r / 20);
                
                pattern[y * size + x] = radial * angular * radialVar;
            }
        }
        
        return this.normalizePattern(pattern);
    }
    
    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }
    
    seededRandom(seed) {
        return function() {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            return seed / 0x7fffffff;
        };
    }
    
    viridisColor(t) {
        const c0 = [68, 1, 84];
        const c1 = [59, 82, 139];
        const c2 = [33, 145, 140];
        const c3 = [94, 201, 98];
        const c4 = [253, 231, 37];
        
        let r, g, b;
        if (t < 0.25) {
            const s = t * 4;
            r = c0[0] + (c1[0] - c0[0]) * s;
            g = c0[1] + (c1[1] - c0[1]) * s;
            b = c0[2] + (c1[2] - c0[2]) * s;
        } else if (t < 0.5) {
            const s = (t - 0.25) * 4;
            r = c1[0] + (c2[0] - c1[0]) * s;
            g = c1[1] + (c2[1] - c1[1]) * s;
            b = c1[2] + (c2[2] - c1[2]) * s;
        } else if (t < 0.75) {
            const s = (t - 0.5) * 4;
            r = c2[0] + (c3[0] - c2[0]) * s;
            g = c2[1] + (c3[1] - c2[1]) * s;
            b = c2[2] + (c3[2] - c2[2]) * s;
        } else {
            const s = (t - 0.75) * 4;
            r = c3[0] + (c4[0] - c3[0]) * s;
            g = c3[1] + (c4[1] - c3[1]) * s;
            b = c3[2] + (c4[2] - c3[2]) * s;
        }
        
        return [Math.round(r), Math.round(g), Math.round(b)];
    }
    
    applyOperation(operation, params) {
        const size = this.canvasSize;
        let newData;
        let opName = '';
        let opParams = '';
        let matrix = SymmetryOps.identity();
        
        switch (operation) {
            case 'rotate':
                const angle = parseFloat(params.angle);
                newData = ImageTransform.rotate(this.transformedImageData, angle, size, size);
                opName = `Rotación`;
                opParams = `${angle}°`;
                matrix = SymmetryOps.rotationMatrix(angle);
                break;
                
            case 'reflect':
                const axis = params.axis;
                newData = ImageTransform.reflect(this.transformedImageData, axis, size, size);
                opName = `Reflexión`;
                opParams = this.translateAxis(axis);
                matrix = SymmetryOps.reflectionMatrix(axis);
                break;
                
            case 'translate':
                const dx = parseFloat(params.dx);
                const dy = parseFloat(params.dy);
                newData = ImageTransform.translate(this.transformedImageData, dx, dy, size, size);
                opName = `Traslación`;
                opParams = `(${dx}a, ${dy}b)`;
                break;
                
            case 'glide':
                const glideAxis = params.axis;
                newData = ImageTransform.glide(this.transformedImageData, glideAxis, size, size);
                opName = `Glide`;
                opParams = this.translateAxis(glideAxis);
                break;
                
            default:
                return;
        }
        
        // Update cumulative matrix for rotation/reflection
        if (operation === 'rotate' || operation === 'reflect') {
            this.currentMatrix = SymmetryOps.multiplyMatrices(matrix, this.currentMatrix);
        }
        
        // Check if the cumulative transformation is a valid symmetry of this group
        // If so, use the original image instead of the transformed one (perfect match!)
        const isValidSymmetry = this.isCurrentTransformASymmetry(operation, params);
        
        if (isValidSymmetry) {
            // Use original image - this is mathematically identical for valid symmetries
            this.transformedImageData = this.originalImageData.slice();
            console.log(`[Symmetry] ${opName} ${opParams} is a valid symmetry - using original image`);
        } else {
            // Use the computed transformation
            this.transformedImageData = newData;
        }
        
        // Update display
        const imageData = this.transformedCtx.createImageData(size, size);
        imageData.data.set(this.transformedImageData);
        this.transformedCtx.putImageData(imageData, 0, 0);
        
        if (operation === 'rotate' || operation === 'reflect') {
            this.updateMatrixDisplay();
        }
        
        // Add to history
        this.operations.push({ name: opName, params: opParams });
        this.updateHistory();
        
        // Update difference visualization and get correlation
        const correlation = this.updateDifference();
        
        // Update Cayley graph for valid symmetries
        if (isValidSymmetry || correlation >= 0.95) {
            this.updateCayleyGraphPosition(`${opName} ${opParams}`);
        }
    }
    
    /**
     * Check if the current cumulative transformation is a valid symmetry of the current group
     */
    isCurrentTransformASymmetry(operation, params) {
        const group = WallpaperGroups[this.currentGroup];
        if (!group.validSymmetries) return false;
        
        // Build the operation description
        let opDesc = '';
        if (operation === 'rotate') {
            const angle = parseFloat(params.angle);
            opDesc = `rotate_${angle}`;
        } else if (operation === 'reflect') {
            opDesc = `reflect_${params.axis}`;
        } else {
            // Translations and glides - check separately
            return false; // For now, only handle point group operations
        }
        
        // Calculate cumulative angle from the matrix
        // For a rotation matrix: [[cos, -sin], [sin, cos]]
        // For a reflection matrix: det = -1
        const m = this.currentMatrix;
        const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        const isReflection = det < 0;
        
        // Get cumulative rotation angle
        let cumulativeAngle = Math.atan2(m[1][0], m[0][0]) * 180 / Math.PI;
        if (cumulativeAngle < 0) cumulativeAngle += 360;
        cumulativeAngle = Math.round(cumulativeAngle) % 360;
        
        // Check if this cumulative state matches any valid symmetry
        for (const sym of group.validSymmetries) {
            // Check rotations
            for (const op of sym.ops) {
                if (op.type === 'rotate') {
                    // For pure rotations, check if cumulative angle matches
                    if (!isReflection && cumulativeAngle === op.angle) {
                        return true;
                    }
                }
                if (op.type === 'reflect') {
                    // For reflections, check if we have a reflection
                    if (isReflection) {
                        // Check the reflection axis matches
                        if (operation === 'reflect' && params.axis === op.axis) {
                            return true;
                        }
                        // For accumulated reflections (odd number), still valid
                        return true;
                    }
                }
            }
            // Identity (no ops) matches when cumulative angle is 0 and no reflection
            if (sym.ops.length === 0 && cumulativeAngle === 0 && !isReflection) {
                return true;
            }
        }
        
        // Additional check: for groups with high rotation order,
        // all multiples of the base angle are valid
        const rotationOrder = group.rotationOrder;
        if (rotationOrder > 1 && !isReflection) {
            const baseAngle = 360 / rotationOrder;
            if (cumulativeAngle % baseAngle === 0) {
                return true;
            }
        }
        
        // For groups with reflections, any reflection is valid if the group has reflection
        if (isReflection && group.hasReflection) {
            return true;
        }
        
        return false;
    }
    
    translateAxis(axis) {
        const translations = {
            'vertical': 'Vertical',
            'horizontal': 'Horizontal',
            'diagonal': 'Diagonal',
            'antidiagonal': 'Anti-diagonal'
        };
        return translations[axis] || axis;
    }
    
    updateDifference() {
        const size = this.canvasSize;
        const diffData = ImageTransform.difference(
            this.originalImageData, 
            this.transformedImageData, 
            size, 
            size
        );
        
        const imageData = this.differenceCtx.createImageData(size, size);
        imageData.data.set(diffData);
        this.differenceCtx.putImageData(imageData, 0, 0);
        
        // Calculate and display correlation
        const correlation = ImageTransform.correlation(
            this.originalImageData, 
            this.transformedImageData
        );
        
        const percent = Math.round(Math.abs(correlation) * 100);
        const correlationValue = document.getElementById('correlationValue');
        correlationValue.textContent = `${percent}%`;
        
        // Color code the correlation
        correlationValue.classList.remove('low', 'medium');
        if (percent >= 98) {
            // EXACT match or very close - it's a symmetry!
            document.querySelector('.canvas-wrapper.difference').classList.add('match-animation');
            setTimeout(() => {
                document.querySelector('.canvas-wrapper.difference').classList.remove('match-animation');
            }, 1000);
        } else if (percent < 50) {
            correlationValue.classList.add('low');
        } else if (percent < 85) {
            correlationValue.classList.add('medium');
        }
        
        // Return correlation for use in Cayley graph updates
        return Math.abs(correlation);
    }
    
    updateHistory() {
        const historyContainer = document.getElementById('operationsHistory');
        
        if (this.operations.length === 0) {
            historyContainer.innerHTML = '<p class="history-empty">No hay operaciones aplicadas</p>';
        } else {
            historyContainer.innerHTML = this.operations.map((op, i) => `
                <div class="history-item">
                    <span class="op-index">${i + 1}.</span>
                    <span class="op-name">${op.name}</span>
                    <span class="op-params">${op.params}</span>
                </div>
            `).join('');
            
            historyContainer.scrollTop = historyContainer.scrollHeight;
        }
        
        document.getElementById('totalOps').textContent = this.operations.length;
    }
    
    reset() {
        this.operations = [];
        this.currentMatrix = SymmetryOps.identity();
        this.currentCayleyNode = 'e';  // Reset to identity
        
        if (this.originalImageData) {
            this.transformedImageData = this.originalImageData.slice();
            
            const size = this.canvasSize;
            const imageData = this.transformedCtx.createImageData(size, size);
            imageData.data.set(this.transformedImageData);
            this.transformedCtx.putImageData(imageData, 0, 0);
            
            this.updateDifference();
        }
        
        this.updateHistory();
        this.updateMatrixDisplay();
        
        // Redraw Cayley graph with identity highlighted
        const group = WallpaperGroups[this.currentGroup];
        this.drawCayleyGraph(group, 'e');
    }
    
    /**
     * Test all Cayley table transitions for the current group.
     * Call from console: app.testCayleyTransitions()
     */
    testCayleyTransitions() {
        const group = WallpaperGroups[this.currentGroup];
        if (!group.cayleyTable || typeof group.cayleyTable.table === 'string') {
            console.log('No Cayley table defined for this group');
            return;
        }
        
        const elements = group.cayleyTable.elements;
        const table = group.cayleyTable.table;
        
        console.log(`\n=== Testing Cayley Table for ${this.currentGroup} ===`);
        console.log('Elements:', elements);
        
        let passed = 0;
        let failed = 0;
        
        // Test: e × x = x for all x (identity property)
        console.log('\n--- Identity property: e × x = x ---');
        for (let i = 0; i < elements.length; i++) {
            const result = table[0][i]; // e is at index 0
            const expected = elements[i];
            if (result === expected) {
                console.log(`✓ e × ${expected} = ${result}`);
                passed++;
            } else {
                console.error(`✗ e × ${expected} = ${result} (expected ${expected})`);
                failed++;
            }
        }
        
        // Test: x × e = x for all x (identity property from right)
        console.log('\n--- Right identity: x × e = x ---');
        for (let i = 0; i < elements.length; i++) {
            const result = table[i][0]; // e is at index 0
            const expected = elements[i];
            if (result === expected) {
                console.log(`✓ ${expected} × e = ${result}`);
                passed++;
            } else {
                console.error(`✗ ${expected} × e = ${result} (expected ${expected})`);
                failed++;
            }
        }
        
        // Test: Each element should have an inverse (x × x⁻¹ = e)
        console.log('\n--- Inverse property: ∃x⁻¹ such that x × x⁻¹ = e ---');
        for (let i = 0; i < elements.length; i++) {
            let hasInverse = false;
            let inverseName = '';
            for (let j = 0; j < elements.length; j++) {
                if (table[i][j] === 'e') {
                    hasInverse = true;
                    inverseName = elements[j];
                    break;
                }
            }
            if (hasInverse) {
                console.log(`✓ ${elements[i]} has inverse: ${inverseName}`);
                passed++;
            } else {
                console.error(`✗ ${elements[i]} has no inverse!`);
                failed++;
            }
        }
        
        // Test closure: all results should be in elements
        console.log('\n--- Closure: all products are in the group ---');
        for (let i = 0; i < elements.length; i++) {
            for (let j = 0; j < elements.length; j++) {
                const result = table[i][j];
                if (elements.includes(result)) {
                    passed++;
                } else {
                    console.error(`✗ ${elements[i]} × ${elements[j]} = "${result}" NOT in elements!`);
                    failed++;
                }
            }
        }
        
        console.log(`\n=== Results: ${passed} passed, ${failed} failed ===\n`);
        
        // Also log the full table for visual inspection
        console.log('Full Cayley table:');
        console.table(table.map((row, i) => {
            const obj = { 'a×b': elements[i] };
            row.forEach((val, j) => obj[elements[j]] = val);
            return obj;
        }));
        
        return { passed, failed };
    }
    
    /**
     * Test all groups' Cayley tables
     * Call from console: app.testAllCayleyTables()
     */
    testAllCayleyTables() {
        const results = {};
        const groups = Object.keys(WallpaperGroups);
        
        for (const groupName of groups) {
            const group = WallpaperGroups[groupName];
            if (!group.cayleyTable || typeof group.cayleyTable.table === 'string') {
                results[groupName] = { skipped: true };
                continue;
            }
            
            const originalGroup = this.currentGroup;
            this.currentGroup = groupName;
            const result = this.testCayleyTransitions();
            this.currentGroup = originalGroup;
            results[groupName] = result;
        }
        
        console.log('\n=== SUMMARY ===');
        for (const [group, result] of Object.entries(results)) {
            if (result.skipped) {
                console.log(`${group}: SKIPPED (no table)`);
            } else if (result.failed === 0) {
                console.log(`${group}: ✓ ALL PASSED (${result.passed})`);
            } else {
                console.error(`${group}: ✗ ${result.failed} FAILED`);
            }
        }
        
        return results;
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new WallpaperExplorer();
});
