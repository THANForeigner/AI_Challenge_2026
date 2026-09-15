#import "cover.typ": cover

#set document(title: "Báo cáo Kỹ thuật - AI Challenge 2026", author: "Đội Thi")
#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 2.5cm, right: 2cm),
  numbering: "1",
  header: align(right)[#text(8pt)[Báo cáo Kỹ thuật AI Challenge TP.HCM 2026]]
)
#set text(font: "Libertinus Serif", size: 11pt, lang: "vi")
#set heading(numbering: "1.1.")
#set par(justify: true)

#cover()

#show outline.entry.where(
  level: 1
): it => {
  v(12pt, weak: true)
  strong(it)
}

#outline(title: [Mục lục], indent: auto)
#pagebreak()

#include "sections/01-overview.typ"
#include "sections/02-pipeline.typ"
#include "sections/03-queries.typ"
#include "sections/04-interface.typ"

