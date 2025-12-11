#!/bin/bash
# 批量渲染架构图

dot -Tpng system_architecture.dot -o system_architecture.png
dot -Tsvg system_architecture.dot -o system_architecture.svg
dot -Tpng feddwa_workflow.dot -o feddwa_workflow.png
dot -Tsvg feddwa_workflow.dot -o feddwa_workflow.svg
dot -Tpng fedclip_architecture.dot -o fedclip_architecture.png
dot -Tsvg fedclip_architecture.dot -o fedclip_architecture.svg
dot -Tpng gprfedsense_architecture.dot -o gprfedsense_architecture.png
dot -Tsvg gprfedsense_architecture.dot -o gprfedsense_architecture.svg

echo '✅ 所有图像已生成!'
