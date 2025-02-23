import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    "",
    {
      text: "Projects",
      icon: "material-symbols:book-2-outline",
      prefix: "demo/",
      link: "demo/",
      children: "structure",
    },
    {
      text: "Posts",
      icon: "material-symbols:add-notes-outline",
      prefix: "Posts/",
      link: "Posts/",
      children: "structure",
      collapsible: false,
    },
    "intro",
  ],
});
