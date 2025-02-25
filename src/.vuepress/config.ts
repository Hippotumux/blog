import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/blog/",

  lang: "en-US",
  title: "Hippotumux's Blog",
  description: "Hippotumux's Blog",
  head: [
    [
      'link',
      {rel: 'icon', href: "Mylogo.jpg"},
    ]
  ],
  port: 7777,
  theme,
  // alias: {
  //   '/': '/blog'  // 訪問根路徑時會重定向到 /blog
  // }
  // 和 PWA 一起启用
  // shouldPrefetch: false,
});
