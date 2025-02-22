# 使用 Node.js 官方映像檔
FROM node:18

# 設置工作目錄
WORKDIR /app

# 複製 package.json 和 package-lock.json
COPY package.json package-lock.json ./

# 安裝依賴（包括 sass-embedded）
RUN npm install
# 安裝 VuePress（穩定版）
RUN npm install -D vuepress@next
RUN npm install -D @vuepress/bundler-webpack@latest

# 安裝 VuePress Theme Hope
RUN npm install -D vuepress-theme-hope@latest

#RUN npm install katex

# 複製專案檔案
COPY . .

# 開放 8080 埠
EXPOSE 7777

# 設定容器啟動時的命令（啟動 VuePress）
CMD ["npm", "run", "docs:dev"]
