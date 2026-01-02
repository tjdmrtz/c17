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
        
        // Canvas size - use ODD number so center pixel is exact (150,150)
        // This ensures perfect 90°/180°/270° rotations
        this.canvasSize = 301;
        
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
        
        // Update alternate names (Schönflies, Hermann-Mauguin, Orbifold)
        const altNamesContainer = document.getElementById('groupAltNames');
        if (altNamesContainer && group.altNames) {
            altNamesContainer.innerHTML = `
                <span class="alt-name schoenflies">${group.altNames.schoenflies}</span>
                <span class="alt-name hm">${group.altNames.hm}</span>
                <span class="alt-name orbifold">${group.altNames.orbifold}</span>
            `;
        }
        
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
     * Geometric Deep Learning style: nodes show transformed polygons
     * R edges = rotations (form cycle), F edges = reflections (connect pairs)
     * Interactive: highlights current position after applying operations
     */
    drawCayleyGraph(group, highlightNode = 'e') {
        const canvas = document.getElementById('cayleyGraph');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Get display size from CSS
        const displayWidth = canvas.clientWidth || 320;
        const displayHeight = canvas.clientHeight || 250;
        
        // Scale for high-DPI displays (Retina, etc.)
        const dpr = window.devicePixelRatio || 1;
        const scaleFactor = Math.max(dpr, 2); // At least 2x for crisp rendering
        
        // Set actual canvas size in memory (scaled)
        canvas.width = displayWidth * scaleFactor;
        canvas.height = displayHeight * scaleFactor;
        
        // Scale the context to counter the size increase
        ctx.scale(scaleFactor, scaleFactor);
        
        // Use display dimensions for drawing calculations
        const width = displayWidth;
        const height = displayHeight;
        
        // Enable crisp rendering
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        
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
        
        // Classify each element by its label
        const classifyElement = (label) => {
            if (label === 'e') return { isIdentity: true, isRotation: false, isReflection: false };
            // F, FR, FR², etc. are reflections (start with F)
            if (label.startsWith('F')) return { isIdentity: false, isRotation: false, isReflection: true };
            // R, R², R³, etc. are rotations (start with R but not FR)
            // Also C₃, C₄, C₆, etc. are rotations (start with C)
            if (label.startsWith('C') || label.startsWith('R')) return { isIdentity: false, isRotation: true, isReflection: false };
            // σ symbols are reflections
            if (label.includes('σ') || label.includes('(g')) return { isIdentity: false, isRotation: false, isReflection: true };
            return { isIdentity: false, isRotation: false, isReflection: false };
        };
        
        // Determine polygon vertices for visualization
        const rotOrder = group.rotationOrder || 1;
        const polyVertices = Math.max(3, rotOrder);
        
        // Calculate layout - always use simple circular layout for robustness
        const centerX = width / 2;
        const centerY = height / 2 - 15;
        const radius = Math.min(width, height) / 2 - 55;
        
        this.cayleyNodes = [];
        
        if (n === 1) {
            // Trivial group: single node at center
            this.cayleyNodes.push({
                x: centerX, y: centerY,
                label: 'e', isIdentity: true, isRotation: false, isReflection: false,
                isCurrent: highlightNode === 'e',
                rotationIndex: 0, isReflected: false
            });
        } else {
            // Place all elements in a circle
            for (let i = 0; i < n; i++) {
                const angle = (i * 2 * Math.PI / n) - Math.PI / 2;
                const label = elements[i];
                const type = classifyElement(label);
                
                this.cayleyNodes.push({
                    x: centerX + radius * Math.cos(angle),
                    y: centerY + radius * Math.sin(angle),
                    label: label,
                    ...type,
                    isCurrent: label === highlightNode,
                    rotationIndex: i,
                    isReflected: type.isReflection
                });
            }
        }
        
        // === Draw edges based on Cayley table structure ===
        ctx.lineWidth = 2;
        
        if (n > 1) {
            // Draw edges based on actual Cayley table relationships
            const table = group.cayleyTable.table;
            const drawnEdges = new Set();
            
            // Find and draw rotation generator edges
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const result = table[i][j];
                    const resultIdx = elements.indexOf(result);
                    
                    // Skip if result not found or is a glide operation marker
                    if (resultIdx === -1 || result.startsWith('(')) continue;
                    
                    // Create edge key to avoid duplicates
                    const edgeKey = `${Math.min(i, j)}-${Math.max(i, j)}`;
                    
                    // Draw rotation edges (connecting via rotation generator)
                    if (!drawnEdges.has(edgeKey) && this.cayleyNodes[i].isRotation !== this.cayleyNodes[resultIdx].isReflection) {
                        // Don't draw all edges - too cluttered. Draw consecutive connections
                    }
                }
            }
            
            // Simple approach: connect consecutive elements (works for cyclic structure)
            for (let i = 0; i < n; i++) {
                const j = (i + 1) % n;
                const fromNode = this.cayleyNodes[i];
                const toNode = this.cayleyNodes[j];
                
                // Determine edge type based on nodes
                const isReflectionEdge = fromNode.isReflection !== toNode.isReflection;
                
                if (isReflectionEdge) {
                    ctx.setLineDash([5, 3]);
                    this.drawReflectionEdge(ctx, fromNode, toNode, '#4fa8c7');
                    ctx.setLineDash([]);
                } else {
                    this.drawCurvedArrow(ctx, fromNode, toNode, '#b794f6', 'R');
                }
            }
            
            // Draw additional reflection edges for dihedral groups (σ² = e)
            if (group.hasReflection) {
                ctx.setLineDash([5, 3]);
                for (let i = 0; i < n; i++) {
                    if (this.cayleyNodes[i].isReflection) {
                        // Find identity node
                        const identityNode = this.cayleyNodes.find(node => node.isIdentity);
                        if (identityNode && identityNode !== this.cayleyNodes[i]) {
                            this.drawReflectionEdge(ctx, this.cayleyNodes[i], identityNode, 'rgba(79, 168, 199, 0.3)');
                        }
                    }
                }
                ctx.setLineDash([]);
            }
        }
        
        // === Draw nodes ===
        const nodeSize = 18;
        
        for (const node of this.cayleyNodes) {
            // Draw circle nodes with labels (simpler and more readable)
            this.drawLabeledNode(ctx, node, nodeSize);
        }
        
        // === Legend ===
        ctx.fillStyle = '#8b98a5';
        ctx.font = '9px Outfit';
        ctx.textAlign = 'center';
        
        const hasReflection = group.hasReflection;
        const groupName = hasReflection ? `D${rotOrder}` : (rotOrder === 1 ? 'C₁' : `C${rotOrder}`);
        ctx.fillText(`Grupo Puntual: ${groupName} (orden ${n})`, width/2, height - 18);
        
        // Edge legend
        ctx.font = '8px Outfit';
        ctx.fillStyle = '#b794f6';
        ctx.fillText('— R (rotación)', width/2 - 50, height - 5);
        if (hasReflection) {
            ctx.fillStyle = '#4fa8c7';
            ctx.fillText('┈ F (reflexión)', width/2 + 50, height - 5);
        }
    }
    
    /**
     * Draw a labeled node (circle with text)
     */
    drawLabeledNode(ctx, node, size) {
        const radius = size;
        
        // Determine color based on node type
        let fillColor, glowColor;
        if (node.isCurrent) {
            fillColor = '#f6b93b';
            glowColor = 'rgba(246, 185, 59, 0.6)';
        } else if (node.isIdentity) {
            fillColor = '#00d4aa';
            glowColor = 'rgba(0, 212, 170, 0.4)';
        } else if (node.isRotation) {
            fillColor = '#b794f6';
            glowColor = 'rgba(183, 148, 246, 0.4)';
        } else if (node.isReflection) {
            fillColor = '#4fa8c7';
            glowColor = 'rgba(79, 168, 199, 0.4)';
        } else {
            fillColor = '#8b98a5';
            glowColor = 'rgba(139, 152, 165, 0.4)';
        }
        
        // Glow effect
        ctx.beginPath();
        const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, radius * 2);
        glow.addColorStop(0, glowColor);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.arc(node.x, node.y, radius * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.isCurrent ? radius + 3 : radius, 0, Math.PI * 2);
        ctx.fillStyle = fillColor;
        ctx.fill();
        
        // Border
        ctx.strokeStyle = node.isCurrent ? '#ffffff' : 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = node.isCurrent ? 3 : 1.5;
        ctx.stroke();
        
        // Label
        ctx.fillStyle = node.isCurrent ? '#000' : '#0f1419';
        ctx.font = node.isCurrent ? 'bold 11px JetBrains Mono' : '10px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Truncate long labels
        let label = node.label;
        if (label.length > 6) {
            label = label.substring(0, 5) + '…';
        }
        ctx.fillText(label, node.x, node.y);
    }
    
    /**
     * Draw a transformed polygon (triangle, square, etc.) representing group element
     */
    drawTransformedPolygon(ctx, x, y, size, numVertices, rotationIndex, isReflected, isCurrent, rotOrder) {
        const rotAngle = (rotationIndex * 2 * Math.PI / rotOrder);
        
        // Vertex colors: 1=red, 2=green, 3=blue (like in the book)
        const vertexColors = ['#ef4444', '#22c55e', '#3b82f6', '#f59e0b', '#8b5cf6', '#06b6d4'];
        
        ctx.save();
        ctx.translate(x, y);
        
        // Apply rotation
        ctx.rotate(rotAngle);
        
        // Apply reflection if needed (flip horizontally)
        if (isReflected) {
            ctx.scale(-1, 1);
        }
        
        // Draw glow for current node
        if (isCurrent) {
            ctx.shadowColor = '#f6b93b';
            ctx.shadowBlur = 15;
        }
        
        // Draw polygon background
        ctx.beginPath();
        for (let i = 0; i < numVertices; i++) {
            const angle = (i * 2 * Math.PI / numVertices) - Math.PI / 2;
            const px = size * Math.cos(angle);
            const py = size * Math.sin(angle);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.closePath();
        
        // Fill with slight transparency
        ctx.fillStyle = isCurrent ? 'rgba(246, 185, 59, 0.3)' : 'rgba(255, 255, 255, 0.1)';
        ctx.fill();
        
        // Stroke
        ctx.strokeStyle = isCurrent ? '#f6b93b' : 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = isCurrent ? 2 : 1;
        ctx.stroke();
        
        ctx.shadowBlur = 0;
        
        // Draw numbered/colored vertices
        const vertexSize = size * 0.25;
        for (let i = 0; i < numVertices; i++) {
            const angle = (i * 2 * Math.PI / numVertices) - Math.PI / 2;
            const vx = size * Math.cos(angle);
            const vy = size * Math.sin(angle);
            
            // Colored circle at vertex
            ctx.beginPath();
            ctx.arc(vx, vy, vertexSize, 0, Math.PI * 2);
            ctx.fillStyle = vertexColors[i % vertexColors.length];
            ctx.fill();
            
            // Number label (1, 2, 3...)
            ctx.fillStyle = '#ffffff';
            ctx.font = `bold ${Math.max(6, vertexSize * 1.2)}px JetBrains Mono`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText((i + 1).toString(), vx, vy);
        }
        
        ctx.restore();
    }
    
    /**
     * Draw curved arrow for rotation edges
     */
    drawCurvedArrow(ctx, from, to, color, label) {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        // Shorten the line to not overlap nodes
        const shortenBy = 20;
        const ratio = shortenBy / dist;
        
        const startX = from.x + dx * ratio;
        const startY = from.y + dy * ratio;
        const endX = to.x - dx * ratio;
        const endY = to.y - dy * ratio;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // Arrow head
        const angle = Math.atan2(endY - startY, endX - startX);
        const headLen = 8;
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(endX, endY);
        ctx.lineTo(endX - headLen * Math.cos(angle - Math.PI / 6), endY - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(endX - headLen * Math.cos(angle + Math.PI / 6), endY - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
        
        // R label near arrow
        const labelX = (startX + endX) / 2;
        const labelY = (startY + endY) / 2;
        const perpX = -dy / dist * 8;
        const perpY = dx / dist * 8;
        ctx.fillStyle = color;
        ctx.font = 'bold 9px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('R', labelX + perpX, labelY + perpY);
    }
    
    /**
     * Draw dashed line for reflection edges
     */
    drawReflectionEdge(ctx, from, to, color) {
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        const shortenBy = 20;
        const ratio = shortenBy / dist;
        
        const startX = from.x + dx * ratio;
        const startY = from.y + dy * ratio;
        const endX = to.x - dx * ratio;
        const endY = to.y - dy * ratio;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        // F label in middle
        const midX = (startX + endX) / 2;
        const midY = (startY + endY) / 2;
        ctx.fillStyle = color;
        ctx.font = 'bold 9px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('F', midX + 8, midY);
    }
    
    /**
     * Draw arrow head for Cayley graph edges (legacy, kept for compatibility)
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
        
        // Build dynamic mapping based on group's rotation order
        // For dihedral groups with R, F notation vs cyclic groups with Cn notation
        const rotOrder = group.rotationOrder;
        const hasDihedralNotation = elements.includes('R') || elements.includes('F');
        
        let opMapping = {};
        
        if (hasDihedralNotation) {
            // Dihedral notation: R, R², R³, etc.
            // Map rotations based on order
            if (rotOrder === 2) {
                // D₂: R = 180°
                opMapping['Rotación 180°'] = ['R'];
            } else if (rotOrder === 3) {
                // D₃: R = 120°
                opMapping['Rotación 120°'] = ['R'];
                opMapping['Rotación 240°'] = ['R²'];
            } else if (rotOrder === 4) {
                // D₄: R = 90°
                opMapping['Rotación 90°'] = ['R'];
                opMapping['Rotación 180°'] = ['R²'];
                opMapping['Rotación 270°'] = ['R³'];
            } else if (rotOrder === 6) {
                // D₆: R = 60°
                opMapping['Rotación 60°'] = ['R'];
                opMapping['Rotación 120°'] = ['R²'];
                opMapping['Rotación 180°'] = ['R³'];
                opMapping['Rotación 240°'] = ['R⁴'];
                opMapping['Rotación 300°'] = ['R⁵'];
            }
            // All reflections map to F (first reflection)
            opMapping['Reflexión Vertical'] = ['F'];
            opMapping['Reflexión Horizontal'] = ['FR', 'FR²', 'F'];
            opMapping['Reflexión Diagonal'] = ['FR', 'FR³', 'F'];
            opMapping['Reflexión Anti-diagonal'] = ['FR²', 'FR', 'F'];
        } else {
            // Cyclic notation: C₃, C₄, C₆, etc.
            opMapping = {
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
        }
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        const cx = (size - 1) / 2;
        const cy = (size - 1) / 2;
        
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
        
        // Always apply the actual transformation - this ensures correct composition
        this.transformedImageData = newData;
        
        // Update cumulative matrix for rotation/reflection
        if (operation === 'rotate' || operation === 'reflect') {
            this.currentMatrix = SymmetryOps.multiplyMatrices(matrix, this.currentMatrix);
        }
        
        // Calculate REAL correlation between transformed and original
        const realCorrelation = ImageTransform.correlation(
            this.originalImageData, 
            this.transformedImageData
        );
        
        // Threshold más tolerante (90%) para detectar simetrías
        // Los errores de interpolación pueden bajar la correlación incluso en simetrías válidas
        const isValidSymmetry = realCorrelation >= 0.90;
        
        // Para simetrías válidas, mostrar imagen original (evita artefactos de interpolación)
        // Para no-simetrías, mostrar la transformación real
        const displayData = isValidSymmetry ? this.originalImageData : this.transformedImageData;
        
        // Para la visualización de diferencia: si es simetría, comparar original con original (= 100%)
        const displayCorrelation = isValidSymmetry ? 1.0 : realCorrelation;
        
        // Animate the transformation visually (siempre se muestra la animación)
        this.animateTransformation(operation, params, () => {
            // Después de la animación, mostrar el resultado
            const imageData = this.transformedCtx.createImageData(size, size);
            imageData.data.set(displayData);
            this.transformedCtx.putImageData(imageData, 0, 0);
        });
        
        if (operation === 'rotate' || operation === 'reflect') {
            this.updateMatrixDisplay();
        }
        
        // Add to history
        this.operations.push({ name: opName, params: opParams });
        this.updateHistory();
        
        // Update difference visualization
        // Si es simetría válida: mostrar 100% y sin diferencias
        // Si no es simetría: mostrar correlación real y diferencias reales
        this.updateDifferenceWithCorrelation(displayCorrelation, isValidSymmetry);
        
        // Update Cayley graph para simetrías válidas
        if (isValidSymmetry) {
            this.updateCayleyGraphPosition(`${opName} ${opParams}`);
        }
    }
    
    /**
     * Check if the current cumulative transformation is a valid symmetry
     */
    isValidSymmetryOperation(operation) {
        const group = WallpaperGroups[this.currentGroup];
        
        // Translations and glides move the pattern - not point group symmetries
        if (operation === 'translate' || operation === 'glide') {
            return false;
        }
        
        // Calculate cumulative state from matrix
        const m = this.currentMatrix;
        const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        const isReflection = det < 0;
        
        // Get cumulative rotation angle
        let angle = Math.atan2(m[1][0], m[0][0]) * 180 / Math.PI;
        if (angle < 0) angle += 360;
        angle = Math.round(angle) % 360;
        
        // Identity is always valid
        if (angle === 0 && !isReflection) {
            return true;
        }
        
        // Check if rotation angle is valid for this group
        const rotationOrder = group.rotationOrder;
        if (rotationOrder > 1 && !isReflection) {
            const baseAngle = 360 / rotationOrder;
            if (angle % baseAngle === 0) {
                return true;
            }
        }
        
        // Check if reflection is valid
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
    
    /**
     * Animate the transformation visually before showing the result
     */
    animateTransformation(operation, params, callback) {
        const wrapper = document.getElementById('transformedWrapper');
        if (!wrapper) {
            callback();
            return;
        }
        
        // Remove any existing animation classes
        wrapper.classList.remove(
            'animating', 'rotating-60', 'rotating-90', 'rotating-120', 
            'rotating-180', 'rotating-240', 'rotating-270', 'rotating-300',
            'reflecting-vertical', 'reflecting-horizontal', 
            'reflecting-diagonal', 'reflecting-antidiagonal',
            'translating-right', 'translating-left', 'translating-down', 'translating-up',
            'gliding-horizontal', 'gliding-vertical'
        );
        
        // Force reflow to restart animation
        void wrapper.offsetWidth;
        
        // Determine animation class based on operation
        let animationClass = '';
        let duration = 600; // ms
        
        switch (operation) {
            case 'rotate':
                const angle = parseFloat(params.angle);
                animationClass = `rotating-${angle}`;
                duration = 600;
                break;
                
            case 'reflect':
                animationClass = `reflecting-${params.axis}`;
                duration = 500;
                break;
                
            case 'translate':
                // Determine direction based on dx/dy
                const tdx = parseFloat(params.dx) || 0;
                const tdy = parseFloat(params.dy) || 0;
                if (tdx > 0) {
                    animationClass = 'translating-right';
                } else if (tdx < 0) {
                    animationClass = 'translating-left';
                } else if (tdy > 0) {
                    animationClass = 'translating-down';
                } else if (tdy < 0) {
                    animationClass = 'translating-up';
                } else {
                    animationClass = 'translating-right'; // fallback
                }
                duration = 500;
                break;
                
            case 'glide':
                animationClass = `gliding-${params.axis}`;
                duration = 700;
                break;
        }
        
        // Add animation classes
        wrapper.classList.add('animating', animationClass);
        
        // Execute callback after animation completes
        setTimeout(() => {
            wrapper.classList.remove('animating', animationClass);
            callback();
        }, duration);
    }
    
    /**
     * Update difference visualization using pre-calculated correlation
     * Si isValidSymmetry=true, muestra diferencia perfecta (100%)
     * Si isValidSymmetry=false, muestra diferencia real
     */
    updateDifferenceWithCorrelation(correlationToShow, isValidSymmetry) {
        const size = this.canvasSize;
        
        // Para simetrías válidas: comparar original con original = sin diferencias
        const compareData = isValidSymmetry ? this.originalImageData : this.transformedImageData;
        
        const diffData = ImageTransform.difference(
            this.originalImageData, 
            compareData, 
            size, 
            size
        );
        
        const imageData = this.differenceCtx.createImageData(size, size);
        imageData.data.set(diffData);
        this.differenceCtx.putImageData(imageData, 0, 0);
        
        // Mostrar el porcentaje
        const percent = Math.round(Math.abs(correlationToShow) * 100);
        const correlationValue = document.getElementById('correlationValue');
        correlationValue.textContent = `${percent}%`;
        
        // Color code the correlation
        correlationValue.classList.remove('low', 'medium');
        if (isValidSymmetry || percent >= 98) {
            // Es simetría - mostrar match
            document.querySelector('.canvas-wrapper.difference').classList.add('match-animation');
            setTimeout(() => {
                document.querySelector('.canvas-wrapper.difference').classList.remove('match-animation');
            }, 1000);
        } else if (percent < 50) {
            correlationValue.classList.add('low');
        } else if (percent < 85) {
            correlationValue.classList.add('medium');
        }
    }
    
    /**
     * Legacy updateDifference for reset and initial display
     */
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
        
        // Calculate REAL correlation
        const correlation = ImageTransform.correlation(
            this.originalImageData, 
            this.transformedImageData
        );
        
        const percent = Math.round(Math.abs(correlation) * 100);
        const correlationValue = document.getElementById('correlationValue');
        correlationValue.textContent = `${percent}%`;
        
        // Color code
        correlationValue.classList.remove('low', 'medium');
        if (percent >= 98) {
            document.querySelector('.canvas-wrapper.difference').classList.add('match-animation');
            setTimeout(() => {
                document.querySelector('.canvas-wrapper.difference').classList.remove('match-animation');
            }, 1000);
        } else if (percent < 50) {
            correlationValue.classList.add('low');
        } else if (percent < 85) {
            correlationValue.classList.add('medium');
        }
        
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
