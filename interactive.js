// ========== micrograd interactive demos ==========
// Every getElementById is guarded with `if (!el) return;` so missing
// elements never throw, regardless of which slide is on screen.

// ---------------------------------------------------------------------------
// Demo 1: Backprop Stepper
// A small expression graph:  e = (a*b) + c ,  L = e * f
// with a=2, b=-3, c=10, f=-2  →  e=-6, L=12
// Clicking "Step backward" walks reverse-mode autodiff one node at a time,
// applying the local derivative (chain rule) and accumulating into .grad.
// ---------------------------------------------------------------------------

const BP_NODES = ['a', 'b', 'c', 'e', 'f', 'L'];

// data values (the forward pass result)
const bpData = { a: 2.0, b: -3.0, c: 10.0, e: -6.0, f: -2.0, L: 12.0 };

// the ordered backward steps, each describing how a node pushes grad to its
// children using the local derivative of its producing op.
const BP_STEPS = [
    {
        node: 'L',
        text: 'Seed the output: dL/dL = 1.0. Backprop always starts by setting the gradient of the final node to 1.',
        apply: g => { g.L = 1.0; },
        touch: ['L'],
    },
    {
        node: 'L',
        text: 'L = e * f. For multiplication the local derivative w.r.t. each input is the OTHER input. So e.grad += f.data * L.grad = (-2)(1) = -2 and f.grad += e.data * L.grad = (-6)(1) = -6.',
        apply: g => { g.e += bpData.f * g.L; g.f += bpData.e * g.L; },
        touch: ['e', 'f'],
    },
    {
        node: 'e',
        text: 'e = (a*b) + c. Addition routes the gradient through unchanged (local derivative = 1), so both the a*b sub-result and c receive e.grad = -2. Here c.grad += 1 * e.grad = -2.',
        apply: g => { g.c += 1.0 * g.e; },
        touch: ['c'],
    },
    {
        node: 'e',
        text: 'The remaining child of e is the product a*b. Multiplication again swaps inputs: a.grad += b.data * e.grad = (-3)(-2) = 6 and b.grad += a.data * e.grad = (2)(-2) = -4.',
        apply: g => { g.a += bpData.b * g.e; g.b += bpData.a * g.e; },
        touch: ['a', 'b'],
    },
];

let bpGrad = {};
let bpStepIdx = 0;

function bpReset() {
    bpGrad = { a: 0, b: 0, c: 0, e: 0, f: 0, L: 0 };
    bpStepIdx = 0;
    bpRender();
    const status = document.getElementById('bp-status');
    if (status) {
        status.innerHTML = '<p>All gradients start at <strong>0</strong>. Click <em>Step backward</em> to seed the output and propagate gradients in reverse topological order.</p>';
    }
}

function bpRender(touched) {
    BP_NODES.forEach(name => {
        const el = document.getElementById('bp-grad-' + name);
        if (el) el.textContent = 'grad ' + bpGrad[name].toFixed(2);
        const card = document.getElementById('bp-node-' + name);
        if (!card) return;
        card.classList.toggle('flow-active', !!(touched && touched.includes(name)));
    });
}

function bpStep() {
    const status = document.getElementById('bp-status');
    if (bpStepIdx >= BP_STEPS.length) {
        if (status) status.innerHTML = '<p><strong>Done.</strong> The graph now holds dL/dx for every node. Compare with PyTorch: identical values. Press <em>Reset</em> to replay.</p>';
        bpRender();
        return;
    }
    const step = BP_STEPS[bpStepIdx];
    step.apply(bpGrad);
    bpRender(step.touch);
    if (status) status.innerHTML = '<p><strong>Step ' + (bpStepIdx + 1) + '/' + BP_STEPS.length + ' — node ' + step.node + ':</strong> ' + step.text + '</p>';
    bpStepIdx++;
}

document.addEventListener('DOMContentLoaded', () => {
    const stepBtn = document.getElementById('bp-step-btn');
    const resetBtn = document.getElementById('bp-reset-btn');
    if (!stepBtn && !resetBtn) return;     // demo not on this deck/slide
    if (stepBtn) stepBtn.addEventListener('click', bpStep);
    if (resetBtn) resetBtn.addEventListener('click', bpReset);
    bpReset();
});

// ---------------------------------------------------------------------------
// Demo 2: Single-Neuron Forward + Backward
// n = tanh(w1*x1 + w2*x2 + b)
// Sliders set x1, x2, w1, w2, b. We show the forward output and the
// gradients of the output w.r.t. every input/weight (local derivatives
// via the chain rule through tanh: d tanh/d z = 1 - tanh(z)^2).
// ---------------------------------------------------------------------------

function neuronUpdate() {
    const ids = ['nx1', 'nx2', 'nw1', 'nw2', 'nb'];
    const v = {};
    for (const id of ids) {
        const el = document.getElementById(id);
        if (!el) return;                  // any missing → bail safely
        v[id] = parseFloat(el.value);
        const out = document.getElementById(id + '-val');
        if (out) out.textContent = v[id].toFixed(2);
    }

    // forward pass
    const z = v.nw1 * v.nx1 + v.nw2 * v.nx2 + v.nb;   // pre-activation
    const o = Math.tanh(z);                            // activation

    // backward pass: do/dz = 1 - tanh(z)^2
    const dz = 1 - o * o;
    const grad = {
        x1: dz * v.nw1,   // do/dx1 = (1-o^2) * w1
        x2: dz * v.nw2,
        w1: dz * v.nx1,   // do/dw1 = (1-o^2) * x1
        w2: dz * v.nx2,
        b:  dz * 1.0,     // do/db  = (1-o^2)
    };

    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };
    set('n-z', z.toFixed(3));
    set('n-out', o.toFixed(3));
    set('n-gx1', grad.x1.toFixed(3));
    set('n-gx2', grad.x2.toFixed(3));
    set('n-gw1', grad.w1.toFixed(3));
    set('n-gw2', grad.w2.toFixed(3));
    set('n-gb', grad.b.toFixed(3));
}

document.addEventListener('DOMContentLoaded', () => {
    const sliders = ['nx1', 'nx2', 'nw1', 'nw2', 'nb'];
    let found = false;
    sliders.forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.addEventListener('input', neuronUpdate); found = true; }
    });
    if (found) neuronUpdate();
});
