import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    "",
    {
      text: "Projects",
      icon: "laptop-code",
      prefix: "demo/",
      link: "demo/",
      children: "structure",
    },
    {
      text: "Posts",
      icon: "book",
      prefix: "Posts/",
      link: "Posts/",
      children: "structure",
      collapsible: false,
    },
    "intro",
  ],
});
