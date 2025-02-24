---
home: false
title: false
navbar: false
sidebar: false
breadcrumb: false
pageInfo: false
contributors: false
editLink: false
lastUpdated: false
prev: false
next: false
comment: false
footer: false
backtotop: false
---


<div class="matrix-rain">
    <canvas ref="matrixCanvas"></canvas>
</div>


<script>
export default {
  name: 'MatrixRain',
  data() {
    return {
      // 你想顯示的字元，可自行替換或增加
      letters: '0123456789!@#$%^&*(_+)',
      fontSize: 16,       // 字體大小
      drops: [],          // 每個垂直字串的「起始位置」陣列
      columns: 0,         // 依照畫布寬度計算能放幾個垂直字串
      ctx: null,          // canvas context
      canvasWidth: 0,
      canvasHeight: 0,
      animationInterval: null, // setInterval 用來控制動畫
    }
  },
  mounted() {
    const canvas = this.$refs.matrixCanvas;
    this.ctx = canvas.getContext('2d');

    // 初始化畫布尺寸
    this.resizeCanvas();
    // 初始化每個垂直字串的起始位置
    this.initDrops();

    // 以約 30FPS (1000/33) 的速度重複呼叫 draw()，產生連續動畫
    this.animationInterval = setInterval(this.draw, 25);

    // 監聽視窗大小改變，及時更新畫布
    window.addEventListener('resize', this.handleResize);
  },
  beforeDestroy() {
    // 組件卸載前，清除動畫與監聽
    clearInterval(this.animationInterval);
    window.removeEventListener('resize', this.handleResize);
  },
  methods: {
    // 調整畫布尺寸符合視窗
    resizeCanvas() {
      const canvas = this.$refs.matrixCanvas;
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      this.canvasWidth = canvas.width;
      this.canvasHeight = canvas.height;

      // 根據畫布寬度與字體大小計算可放幾個垂直字串
      this.columns = Math.floor(this.canvasWidth / this.fontSize);
    },
    // 初始化 drops，令每個字串從第一行開始
    initDrops() {
      this.drops = [];
      for (let i = 0; i < this.columns; i++) {
        this.drops[i] = 1;
      }
    },
    // 每次重繪畫面
    draw() {
      // 使用半透明黑色作為背景，營造字尾漸淡的效果
      this.ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
      this.ctx.fillRect(0, 0, this.canvasWidth, this.canvasHeight);

      // 設定字體樣式（綠色字、等寬字體）
      this.ctx.fillStyle = '#0F0';
      this.ctx.font = `${this.fontSize}px monospace`;

      // 逐列繪製
      for (let i = 0; i < this.drops.length; i++) {
        // 隨機從 letters 中取出一個字元
        const text = this.letters.charAt(Math.floor(Math.random() * this.letters.length));
        // 在 (x, y) 位置繪製字元
        this.ctx.fillText(text, i * this.fontSize, this.drops[i] * this.fontSize);

        // 如果超過畫布底部且有一定機率，重頭開始
        if (this.drops[i] * this.fontSize > this.canvasHeight && Math.random() > 0.975) {
          this.drops[i] = 0;
        }
        this.drops[i]++;
      }
    },
    // 監聽視窗改變大小後，重新設定畫布和 drops
    handleResize() {
      this.resizeCanvas();
      this.initDrops();
    },
  },
}
</script>

<style scoped>
.matrix-rain {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  margin: 0;
  padding: 0;
  background: #000;
  overflow: hidden;
  z-index: 9999; /* 在背景層，如需蓋住全部可改成 9999 */
}


canvas {
  display: block;
  width: 100%;
  height: 100%;
}
</style>
