import os
from dotenv import load_dotenv
from tqdm import trange
import re
import pandas as pd

import asyncio
import json

from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Provide an in-depth explanation of the entity’s attributes and activities.
  **(Include all known attributes, actions, behaviors, relationships with other entities or events, and any context from the text that helps describe the entity in a comprehensive manner. Ensure that every relevant detail from the input text is incorporated into the appropriate entity_descriptions, so that no portion of the text is left out. If there are additional facts, observations, or details in the text that do not appear to fit neatly under any existing entity, either create a new entity for them or expand an existing entity’s description to capture them.)**

Format each entity as ("entity"{tuple_delimiter}"<entity_name>"{tuple_delimiter}"<entity_type>"{tuple_delimiter}"<entity_description>")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"{tuple_delimiter}"<source_entity>"{tuple_delimiter}"<target_entity>"{tuple_delimiter}"<relationship_description>"{tuple_delimiter}"<relationship_strength>")

3. Return output in Vietnamese as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

5. Double-check that every detail from the input text is represented in either an entity description or a relationship description. If any piece of information is missing, revise the entity descriptions or create new entities/relationships until you have accounted for all details.

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
3. Ghi nhận thời gian làm việc

Làm việc tại văn phòng:

Công ty sử dụng công cụ xử lý dữ liệu hệ thống bằng thẻ từ, sau đây gọi tắt là máy chấm công để theo dõi thời gian ra, vào nơi làm việc của CBNV.

Mỗi CBNV được cung cấp 01 thẻ từ để ghi lại thời gian làm việc tại Công ty.

CBNV phải quẹt thẻ tại máy chấm công khi bắt đầu làm việc và trước khi ra về. Hệ thống cho phép đồng bộ dữ liệu quẹt thẻ ở toàn bộ văn phòng của Sun*. Trường hợp đi công tác hoặc di chuyển giữa các văn phòng, CBNV đều có thể sử dụng thẻ nhân viên của mình để quẹt và hệ thống ghi nhận giờ quẹt thẻ đầu tiên làm giờ in và giờ quẹt thẻ cuối cùng làm giờ out.

CBNV chỉ được phép dùng thẻ của mình để quẹt thẻ chấm công, không được quẹt thẻ hộ người khác. Trường hợp bị phát hiện quẹt thẻ chấm công hộ người khác, thì người quẹt hộ và người nhờ quẹt hộ sẽ bị áp dụng hình thức khiển trách nội bộ và bị giảm trừ 1.000.000 VNĐ cho mỗi lần vi phạm vào kỳ thưởng hiệu suất giữa năm hoặc thưởng tháng lương thứ 13.

Trường hợp quên thẻ, CBNV làm đơn trên hệ thống và xin xác nhận của CBQL ngay trong ngày phát sinh quên thẻ. CBQL có trách nhiệm xác nhận và phê duyệt đơn ngay sau thời điểm nhận được thông báo từ CBNV.

Trường hợp quên quẹt thẻ, CBNV làm đơn trên hệ thống và xin xác nhận của CBQL trong 03 ngày làm việc kể từ ngày quên quẹt thẻ.

Các trường hợp quên thẻ/quên quẹt thẻ mà không có sự xác nhận từ CBQL theo quy định dẫn đến tình trạng bị thiếu dữ liệu, CBNV sẽ bị tính nghỉ không lương cả ngày hoặc nửa ngày tuỳ vào dữ liệu chấm công thực tế.
################
Output:
("entity"{tuple_delimiter}"CÔNG TY"{tuple_delimiter}"organization"{tuple_delimiter}"Công ty sử dụng công cụ xử lý dữ liệu hệ thống bằng thẻ từ để theo dõi thời gian làm việc của cán bộ nhân viên (CBNV). Công ty áp dụng quy định về việc sử dụng thẻ từ để đảm bảo quản lý chặt chẽ thời gian ra vào làm việc và tính toán các trường hợp vi phạm liên quan đến thời gian làm việc."){record_delimiter}
("entity"{tuple_delimiter}"CBNV"{tuple_delimiter}"person"{tuple_delimiter}"Cán bộ nhân viên (CBNV) là đối tượng thực hiện các quy định về việc sử dụng thẻ từ để ghi nhận thời gian làm việc. CBNV có trách nhiệm quẹt thẻ tại máy chấm công khi bắt đầu làm việc và ra về. CBNV cũng phải chịu các chế tài nếu vi phạm quy định liên quan."){record_delimiter}
("entity"{tuple_delimiter}"MÁY CHẤM CÔNG"{tuple_delimiter}"technology"{tuple_delimiter}"Máy chấm công là công cụ được sử dụng để ghi nhận thời gian ra vào nơi làm việc của CBNV bằng cách quẹt thẻ từ. Máy này đồng bộ dữ liệu toàn bộ văn phòng và hỗ trợ việc ghi nhận dữ liệu chính xác."){record_delimiter}
("entity"{tuple_delimiter}"THẺ TỪ"{tuple_delimiter}"technology"{tuple_delimiter}"Thẻ từ là công cụ cá nhân được cấp cho mỗi CBNV để ghi lại thời gian làm việc tại Công ty. Thẻ từ được sử dụng khi quẹt tại máy chấm công và đảm bảo tính cá nhân hóa trong việc theo dõi thời gian làm việc."){record_delimiter}
("entity"{tuple_delimiter}"CBQL"{tuple_delimiter}"person"{tuple_delimiter}"Cán bộ quản lý (CBQL) có trách nhiệm xác nhận và phê duyệt đơn xin quên thẻ/quên quẹt thẻ của CBNV. CBQL đóng vai trò quan trọng trong việc xác nhận thông tin liên quan đến các trường hợp phát sinh lỗi trong ghi nhận thời gian làm việc."){record_delimiter}
("entity"{tuple_delimiter}"VĂN PHÒNG SUN*"{tuple_delimiter}"location"{tuple_delimiter}"Địa điểm làm việc nơi hệ thống máy chấm công và thẻ từ được áp dụng. Văn phòng hỗ trợ việc đồng bộ dữ liệu quẹt thẻ trên toàn hệ thống của Sun*."){record_delimiter}
("entity"{tuple_delimiter}"QUY ĐỊNH VỀ THẺ TỪ"{tuple_delimiter}"mission"{tuple_delimiter}"Quy định về việc sử dụng thẻ từ yêu cầu CBNV phải quẹt thẻ tại máy chấm công khi bắt đầu làm việc và trước khi ra về. Vi phạm quy định này sẽ dẫn đến chế tài kỷ luật và giảm thưởng."){record_delimiter}
("entity"{tuple_delimiter}"CHẾ TÀI VI PHẠM"{tuple_delimiter}"mission"{tuple_delimiter}"Công ty áp dụng chế tài nghiêm khắc đối với các vi phạm liên quan đến việc sử dụng thẻ từ, bao gồm khiển trách nội bộ và giảm trừ 1.000.000 VNĐ vào kỳ thưởng hiệu suất giữa năm hoặc thưởng lương tháng 13."){record_delimiter}
("entity"{tuple_delimiter}"ĐƠN XIN QUÊN THẺ"{tuple_delimiter}"technology"{tuple_delimiter}"Đơn xin được thực hiện trên hệ thống khi CBNV quên thẻ. CBNV cần xin xác nhận của CBQL ngay trong ngày phát sinh lỗi."){record_delimiter}
("entity"{tuple_delimiter}"ĐƠN XIN QUÊN QUẸT THẺ"{tuple_delimiter}"technology"{tuple_delimiter}"Đơn được thực hiện trên hệ thống khi CBNV quên quẹt thẻ. Cần được nộp và xác nhận bởi CBQL trong vòng 03 ngày làm việc kể từ ngày phát sinh lỗi."){record_delimiter}
("entity"{tuple_delimiter}"DỮ LIỆU CHẤM CÔNG"{tuple_delimiter}"technology"{tuple_delimiter}"Hệ thống dữ liệu ghi nhận giờ vào và giờ ra của CBNV, được đồng bộ trên toàn bộ văn phòng. Các lỗi phát sinh không có xác nhận của CBQL sẽ dẫn đến tình trạng bị tính nghỉ không lương."){record_delimiter}
("relationship"{tuple_delimiter}"CÔNG TY"{tuple_delimiter}"MÁY CHẤM CÔNG"{tuple_delimiter}"Công ty sử dụng máy chấm công để theo dõi thời gian ra vào làm việc của CBNV, hỗ trợ quản lý hệ thống dữ liệu chấm công."{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"THẺ TỪ"{tuple_delimiter}"Mỗi CBNV được cung cấp một thẻ từ để ghi lại thời gian làm việc và phải sử dụng thẻ này cho việc quẹt thẻ chấm công."{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"THẺ TỪ"{tuple_delimiter}"MÁY CHẤM CÔNG"{tuple_delimiter}"Thẻ từ được sử dụng tại máy chấm công để ghi nhận giờ làm việc của CBNV, với dữ liệu được đồng bộ toàn hệ thống."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"CBQL"{tuple_delimiter}"CBQL có trách nhiệm xác nhận đơn từ liên quan đến các trường hợp quên thẻ/quên quẹt thẻ của CBNV, đảm bảo dữ liệu được cập nhật chính xác."{tuple_delimiter}"7"){record_delimiter}
("relationship"{tuple_delimiter}"CBQL"{tuple_delimiter}"CÔNG TY"{tuple_delimiter}"CBQL thực hiện vai trò quản lý, đảm bảo thực hiện các quy định của Công ty về việc ghi nhận thời gian làm việc."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"VĂN PHÒNG SUN*"{tuple_delimiter}"THẺ TỪ"{tuple_delimiter}"Thẻ từ có thể sử dụng tại tất cả văn phòng Sun* để đồng bộ dữ liệu quẹt thẻ giữa các văn phòng."{tuple_delimiter}"7"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"QUY ĐỊNH VỀ THẺ TỪ"{tuple_delimiter}"CBNV phải tuân thủ quy định sử dụng thẻ từ, bao gồm việc quẹt thẻ đúng quy định khi bắt đầu làm việc và trước khi ra về."{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"QUY ĐỊNH VỀ THẺ TỪ"{tuple_delimiter}"CHẾ TÀI VI PHẠM"{tuple_delimiter}"Vi phạm quy định về sử dụng thẻ từ sẽ dẫn đến áp dụng các chế tài kỷ luật nghiêm khắc."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"ĐƠN XIN QUÊN THẺ"{tuple_delimiter}"CBNV phải thực hiện đơn xin quên thẻ trên hệ thống khi xảy ra tình trạng quên mang thẻ từ."{tuple_delimiter}"7"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"ĐƠN XIN QUÊN QUẸT THẺ"{tuple_delimiter}"CBNV có trách nhiệm làm đơn xin xác nhận quên quẹt thẻ trong vòng 03 ngày làm việc để tránh bị tính nghỉ không lương."{tuple_delimiter}"7"){record_delimiter}
("relationship"{tuple_delimiter}"CBQL"{tuple_delimiter}"ĐƠN XIN QUÊN THẺ"{tuple_delimiter}"CBQL chịu trách nhiệm xác nhận và phê duyệt đơn xin quên thẻ của CBNV trong ngày phát sinh."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"CBQL"{tuple_delimiter}"ĐƠN XIN QUÊN QUẸT THẺ"{tuple_delimiter}"CBQL có trách nhiệm xác nhận đơn xin quên quẹt thẻ của CBNV trong vòng 03 ngày làm việc."{tuple_delimiter}"8"){record_delimiter}
("relationship"{tuple_delimiter}"DỮ LIỆU CHẤM CÔNG"{tuple_delimiter}"MÁY CHẤM CÔNG"{tuple_delimiter}"Dữ liệu chấm công được ghi nhận từ máy chấm công, đảm bảo đồng bộ trên toàn hệ thống văn phòng."{tuple_delimiter}"9"){record_delimiter}
("relationship"{tuple_delimiter}"DỮ LIỆU CHẤM CÔNG"{tuple_delimiter}"CBNV"{tuple_delimiter}"Dữ liệu chấm công phản ánh giờ vào và ra của CBNV và là cơ sở tính toán thời gian làm việc."{tuple_delimiter}"9"){record_delimiter}{completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
Điều 5: Quy định về làm thêm giờ

- CBNV phát sinh làm thêm giờ hoặc làm việc tự nguyện ngày thứ Bảy phải điền vào Đơn yêu cầu làm thêm ngoài giờ/Làm việc ngày thứ Bảy trên hệ thống quản lý thời gian làm việc WSM, đồng thời phải được sự phê duyệt của CBQL trước thời hạn chốt công hàng tháng.

- Tổng thời gian làm thêm giờ không quá 40 giờ/tháng và 200 giờ/năm hoặc không quá 300 giờ trong một năm trong các trường hợp theo quy định của pháp luật lao động.

- Cách thức ghi nhận thời gian làm thêm giờ trong tháng được dựa trên hai cơ sở dữ liệu:
    - Thời gian theo đơn Yêu cầu làm thêm giờ được sự đồng ý của CBQL.
    - Thời gian quẹt thẻ thực tế: dựa vào dữ liệu quẹt thẻ thực tế được ghi nhận trên hệ thống wsm.

- Nếu thời gian đăng ký trong đơn lớn hơn thời gian quẹt thẻ thực tế: hệ thống ghi nhận thời gian OT là thời gian quẹt thẻ thực tế.

- Nếu thời gian đăng ký trong đơn nhỏ hơn thời gian quẹt thẻ thực tế: hệ thống ghi nhận thời gian OT là thời gian đăng ký trong đơn. CBNV có trách nhiệm bổ sung đơn đăng ký cho khoảng thời gian chênh lệch trên, đồng thời đơn này cũng phải được sự phê duyệt của CBQL.

- Trong trường hợp phát sinh OT vào ngày cuối cùng của tháng (ngày 23) thì CBNV phải hoàn thiện đơn đăng ký có xác nhận của CBQL trên hệ thống chậm nhất vào 23:59’ ngày hôm đó.

- Nếu thời gian OT kéo dài qua đêm, sẽ phải thực hiện làm hai đơn đăng ký cho hai ngày.

- Thời gian làm thêm với các ngày từ thứ Hai đến thứ Sáu được tính sau 01 giờ kể từ khi kết thúc ca làm việc (muộn nhất là 18:45).
#############
Output:
("entity"{tuple_delimiter}"CBNV"{tuple_delimiter}"person"{tuple_delimiter}"CBNV là viết tắt của cán bộ nhân viên, đại diện cho những người lao động trong tổ chức thực hiện các nhiệm vụ, có trách nhiệm đăng ký làm thêm giờ theo đúng quy định của công ty và pháp luật."){record_delimiter}
("entity"{tuple_delimiter}"CBQL"{tuple_delimiter}"person"{tuple_delimiter}"CBQL là viết tắt của cán bộ quản lý, là người có thẩm quyền phê duyệt các yêu cầu làm thêm giờ của CBNV và đảm bảo tuân thủ các quy định quản lý thời gian làm việc."){record_delimiter}
("entity"{tuple_delimiter}"WSM"{tuple_delimiter}"technology"{tuple_delimiter}"WSM là hệ thống quản lý thời gian làm việc, được sử dụng để ghi nhận và quản lý thời gian làm thêm giờ của CBNV thông qua việc đăng ký và quẹt thẻ thực tế."){record_delimiter}
("entity"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"mission"{tuple_delimiter}"Thời gian làm thêm giờ đề cập đến khoảng thời gian làm việc vượt ngoài giờ hành chính, phải được đăng ký và phê duyệt theo quy định cụ thể, không vượt quá giới hạn pháp luật (40 giờ/tháng, 200 hoặc 300 giờ/năm)."){record_delimiter}
("entity"{tuple_delimiter}"Quẹt thẻ thực tế"{tuple_delimiter}"technology"{tuple_delimiter}"Quẹt thẻ thực tế là hành động sử dụng thẻ để ghi nhận thời gian làm việc thực tế của CBNV trên hệ thống WSM, là cơ sở để đối chiếu với thời gian đăng ký làm thêm giờ."){record_delimiter}
("entity"{tuple_delimiter}"Pháp luật lao động"{tuple_delimiter}"organization"{tuple_delimiter}"Pháp luật lao động là hệ thống quy định pháp lý điều chỉnh các vấn đề liên quan đến lao động, bao gồm giới hạn thời gian làm thêm giờ và các quy định liên quan đến việc làm thêm."){record_delimiter}
("entity"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"technology"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ là tài liệu được CBNV sử dụng để đăng ký thời gian làm thêm giờ trên hệ thống WSM, phải được sự phê duyệt của CBQL trước thời hạn chốt công."){record_delimiter}
("entity"{tuple_delimiter}"Ngày thứ Bảy"{tuple_delimiter}"mission"{tuple_delimiter}"Ngày thứ Bảy là thời gian không thuộc ngày làm việc chính thức trong tuần, được quy định cho việc làm việc tự nguyện hoặc làm thêm giờ, cần đăng ký và được phê duyệt."){record_delimiter}
("entity"{tuple_delimiter}"Thời gian OT kéo dài qua đêm"{tuple_delimiter}"mission"{tuple_delimiter}"Thời gian OT kéo dài qua đêm là khoảng thời gian làm thêm giờ liên tục từ ngày này sang ngày khác, đòi hỏi phải thực hiện hai đơn đăng ký riêng biệt cho từng ngày."){record_delimiter}
("entity"{tuple_delimiter}"Ngày 23"{tuple_delimiter}"time"{tuple_delimiter}"Ngày 23 được quy định là ngày cuối cùng của tháng, là thời điểm cuối cùng để CBNV hoàn tất việc đăng ký làm thêm giờ và được CBQL phê duyệt trước hạn chốt công."){record_delimiter}
("entity"{tuple_delimiter}"Thời gian 01 giờ"{tuple_delimiter}"time"{tuple_delimiter}"Thời gian 01 giờ là khoảng thời gian đệm sau khi ca làm việc kết thúc, từ đó mới bắt đầu tính thời gian làm thêm giờ từ thứ Hai đến thứ Sáu."){record_delimiter}
("entity"{tuple_delimiter}"18:45"{tuple_delimiter}"time"{tuple_delimiter}"18:45 là thời gian muộn nhất được quy định để tính giờ làm thêm trong ngày làm việc bình thường (thứ Hai đến thứ Sáu)."){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"CBQL"{tuple_delimiter}"CBNV phải được sự phê duyệt của CBQL khi đăng ký làm thêm giờ, thể hiện mối quan hệ quản lý và giám sát giữa nhân viên và cấp quản lý."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"WSM"{tuple_delimiter}"CBNV sử dụng hệ thống WSM để đăng ký và ghi nhận thời gian làm thêm giờ, cho thấy mối quan hệ tương tác giữa người lao động và công nghệ quản lý thời gian."{tuple_delimiter}"5"){record_delimiter}
("relationship"{tuple_delimiter}"WSM"{tuple_delimiter}"Quẹt thẻ thực tế"{tuple_delimiter}"WSM ghi nhận thời gian làm thêm giờ dựa trên dữ liệu từ quẹt thẻ thực tế, thể hiện mối liên hệ trực tiếp giữa công nghệ và hành động thực tế."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"Pháp luật lao động"{tuple_delimiter}"Giới hạn thời gian làm thêm giờ được quy định bởi pháp luật lao động, thể hiện sự điều chỉnh và kiểm soát của pháp luật đối với hành vi làm thêm giờ."{tuple_delimiter}"5"){record_delimiter}
("relationship"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"CBQL"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ phải được CBQL phê duyệt, thể hiện sự phụ thuộc của tài liệu vào quyết định của cấp quản lý."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"WSM"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ được nộp và quản lý trên hệ thống WSM, cho thấy sự tích hợp giữa tài liệu và công nghệ quản lý."{tuple_delimiter}"5"){record_delimiter}
("relationship"{tuple_delimiter}"CBNV"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"CBNV phải điền vào đơn yêu cầu làm thêm giờ khi phát sinh làm việc ngoài giờ, thể hiện sự tuân thủ quy định của nhân viên với hệ thống quản lý."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Quẹt thẻ thực tế"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"Thời gian quẹt thẻ thực tế là một trong hai cơ sở để ghi nhận thời gian làm thêm giờ, thể hiện sự liên kết giữa hành động thực tế và thời gian làm thêm được tính toán."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Ngày thứ Bảy"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"Khi làm việc vào ngày thứ Bảy, CBNV cần điền đơn yêu cầu làm thêm giờ và được CBQL phê duyệt, thể hiện mối liên hệ giữa quy định làm việc và tài liệu quản lý."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Thời gian OT kéo dài qua đêm"{tuple_delimiter}"Đơn yêu cầu làm thêm giờ"{tuple_delimiter}"Thời gian OT kéo dài qua đêm đòi hỏi phải thực hiện hai đơn đăng ký riêng biệt, liên kết chặt chẽ với quy định tài liệu."{tuple_delimiter}"5"){record_delimiter}
("relationship"{tuple_delimiter}"Ngày 23"{tuple_delimiter}"CBNV"{tuple_delimiter}"CBNV phải hoàn thành đơn yêu cầu OT trước 23:59 ngày 23, thể hiện mối quan hệ giữa thời gian quy định và trách nhiệm của nhân viên."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Ngày 23"{tuple_delimiter}"CBQL"{tuple_delimiter}"CBQL cần phê duyệt đơn OT của CBNV trước 23:59 ngày 23, cho thấy vai trò quản lý trong kiểm soát thời gian làm thêm."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"Thời gian 01 giờ"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"Thời gian 01 giờ là thời gian chờ bắt đầu tính làm thêm giờ, được quy định cụ thể trong phạm vi thời gian làm thêm giờ."{tuple_delimiter}"4"){record_delimiter}
("relationship"{tuple_delimiter}"18:45"{tuple_delimiter}"Thời gian làm thêm giờ"{tuple_delimiter}"18:45 là thời gian muộn nhất để tính thời gian làm thêm trong ngày làm việc, liên kết với quy định về giờ làm thêm."{tuple_delimiter}"4"){record_delimiter}{completion_delimiter}
#############################

-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""
CONTINUE_PROMPT = "SOME other entities and relationship were missed in the last extraction.  Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities may have still been missed.  Answer {tuple_delimiter}YES{tuple_delimiter} if there are still entities that need to be added else {tuple_delimiter}NO{tuple_delimiter} \n"


DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


ENTITY_TYPES = [
    "System",
    "Asset",
    "Service",
    "Product",
    "Code",
    "Standard",
    "Organization",
    "Allowance",
    "Process",
    "Time",
    "Location",
    "Person",
    "Section",
    "Regulation",
    "Role",
    "Event",
    "Department",
    "Benefit",
    "Instruments",
    "Form",
]

load_dotenv()

model = Agent(
    OpenAIModel(
        "gpt-4o-mini",
        openai_client=AsyncAzureOpenAI(
            api_key=os.environ["GPT4O__KEY"],
            azure_endpoint=os.environ["GPT4O__ENDPOINT"],
            api_version=os.environ["GPT4O__API_VERSION"],
        ),
    )
)


async def extract(text: str, global_context: str) -> str:
    res = await model.run(
        GRAPH_EXTRACTION_PROMPT.format(
            **{
                "entity_types": ENTITY_TYPES,
                "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
                "completion_delimiter": DEFAULT_COMPLETION_DELIMITER,
                "record_delimiter": DEFAULT_RECORD_DELIMITER,
                "global_context": global_context,
                "input_text": text,
            }
        )
    )

    result = res.data

    history = res.new_messages()

    for _ in trange(5, desc="Gleaning...", leave=False):
        glean_res = await model.run(CONTINUE_PROMPT, message_history=history)
        history.extend(glean_res.new_messages())
        result += glean_res.data

        continuation = await model.run(
            LOOP_PROMPT.format(
                **{
                    "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
                }
            )
        )

        if f"{DEFAULT_TUPLE_DELIMITER}NO{DEFAULT_TUPLE_DELIMITER}" in continuation.data:
            break
    return result


def extract_relationship(
    text: str, global_context: str, extracted_data: str
) -> pd.DataFrame:
    output = []
    pattern = r'\("relationship"<\|>"(.*?)"<\|>"(.*?)"<\|>"(.*?)"<\|>"(.*?)"\)'
    matches = re.findall(pattern, extracted_data)
    for match in matches:
        source_entity, target_entity, description, relationship_num = match
        if relationship_num.isnumeric():
            output.append(
                {
                    "text": text,
                    "global_context": global_context,
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "description": description,
                    "relationship_strength": relationship_num,
                }
            )
    return pd.DataFrame(output)


def extract_entity(text: str, global_context: str, extracted_data: str) -> pd.DataFrame:
    outputs = []
    pattern = r'\("entity"<\|>"(.*?)"<\|>"(.*?)"<\|>"(.*?)"\)'
    matches = re.findall(pattern, extracted_data)
    for match in matches:
        entity, type, description = match
        outputs.append(
            {
                "text": text,
                "global_context": global_context,
                "entity": entity,
                "type": type,
                "description": description,
            },
        )
    return pd.DataFrame(outputs)


async def run(text: str, global_context: str, output_dir: str, fname: str):
    t = await extract(text, global_context)
    entities_df = extract_entity(text, global_context, t)
    relationships_df = extract_relationship(text, global_context, t)
    os.makedirs(output_dir, exist_ok=True)
    entities_df.to_excel(
        os.path.join(output_dir, f"{fname}_entities.xlsx"), "entities", index=False
    )
    relationships_df.to_excel(
        os.path.join(output_dir, f"{fname}_relationships.xlsx"),
        "relationships",
        index=False,
    )


fname = "output_chunk_full_ga"
with open(f"{fname}.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# data = {
#     "chunks": [
#         """Quy tắc đặt tiêu đề email theo tài liệu **1-208_Quy Tắc Đặt Tiêu Đề Email (Rev3_14022021)** bao gồm:
# - Tiêu đề email phải viết bằng chữ in hoa, có dấu.
# - Sử dụng tên phòng viết tắt trong tiêu đề: SUN*VN cho toàn công ty, SUN*HN cho Hà Nội, SUN*DN cho Đà Nẵng, SUN*HCM cho Hồ Chí Minh.
# - Email gửi cho nhân viên trong nước có thể bỏ qua nội dung tiếng Anh.
# - Email gửi khách hàng, đối tác: tiêu đề theo mẫu [SUN*VN] NỘI DUNG EMAIL TIẾNG VIỆT/NỘI DUNG EMAIL TIẾNG ANH; bỏ qua tiếng Anh nếu đối tượng là Việt Nam.
# - Email dự án: tiêu đề theo mẫu [TÊN DỰ ÁN] NỘI DUNG EMAIL TIẾNG VIỆT/NỘI DUNG EMAIL TIẾNG ANH.
# - Email cho mail list có người nước ngoài cần tiêu đề và nội dung bằng tiếng Anh."""
#     ],
#     "metadata": [{"global_context": ""}],
# }

# for i in trange(len(data["chunks"]), desc="Extracting chunk"):
#     chunk = data["chunks"][i]
#     gb_context = data["metadata"][i].get("global_context", "")
#     asyncio.run(run(chunk, gb_context, f"output_dir/{fname}", fname))
chunk = """
|**QUY ĐỊNH CHUNG<br>Áp dụng với tất cả các thiết bị được sử dụng trong công việc**|**QUY ĐỊNH CHUNG<br>Áp dụng với tất cả các thiết bị được sử dụng trong công việc**|**THIẾT BỊ CÔNG TY/ KHÁCH HÀNG**|**THIẾT BỊ CÔNG TY/ KHÁCH HÀNG**|**THIẾT BỊ CÁ NHÂN**|**THIẾT BỊ CÁ NHÂN**|
|---|---|---|---|---|---|
|[1. Quản lý thiết bị](None)|[1. Quản lý thiết bị](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|[2. Quản lý truy cập](None)|[2. Quản lý truy cập](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|[3. Thiết lập và cài đặt thiết bị](None)|[3. Thiết lập và cài đặt thiết bị](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|[4. Cài đặt và sử dụng phần mềm](None)|[4. Cài đặt và sử dụng phần mềm](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|[5. Xử lý sự cố](None)|[5. Xử lý sự cố](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|**1. Quản lý thiết bị**|**1.1 Các request liên quan đến thiết bị**|[- Mọi yêu cầu liên quan đến thiết bị do Công ty hoặc Khách hàng cung cấp (đổi, trả, cấp mới, nâng cấp, thay thế linh kiện, sửa chữa...) đều phải được thực hiện thông qua hệ thống quản lý tài sản của Công ty - Sun*Asset và cần có sự cho phép của PL/LM.<br>- Với thiết bị do Khách hàng cung cấp, mọi nhu cầu liên quan đến việc nâng cấp, thay thế linh kiện, sửa chữa... đều phải thông báo và nhận được sự cho phép của Khách hàng trước khi thực hiện.<br>- Trường hợp mang thiết bị ra khỏi phạm vi công ty (bao gồm cả trường hợp đi công tác), nhân viên có trách nhiệm hoàn tất thủ tục Request - Mang thiết bị ra bên ngoài phạm vi công ty và tuân thủ các nội dung trong Cam kết bảo mật khi mang thiết bị ra bên ngoài phạm vi công ty <br>+ Thời hạn sử dụng request: <br>          . Staff: Tối đa 3 tháng<br>          . PL/LM: Tối đa 6 tháng<br>          . Head/SDM/UM: Vô thời hạn<br>+ Kết thúc thời hạn sử dụng, nhân viên không được phép tiếp tục mang thiết bị ra ngoài phạm vi công ty. Trường hợp muốn tiếp tục mang thiết bị ra ngoài phạm vi công ty, nhân viên cần tạo lại Request trước ngày hết hạn.<br>- Khi thay đổi thiết bị hoặc mang thêm thiết bị khác ra khỏi phạm vi công ty (bao gồm cả trường hợp đi công tác) cần tạo Request mới và ký lại Cam kết bảo mật](https://asset.sun-asterisk.vn/my_requests)|[- Mọi yêu cầu liên quan đến thiết bị do Công ty hoặc Khách hàng cung cấp (đổi, trả, cấp mới, nâng cấp, thay thế linh kiện, sửa chữa...) đều phải được thực hiện thông qua hệ thống quản lý tài sản của Công ty - Sun*Asset và cần có sự cho phép của PL/LM.<br>- Với thiết bị do Khách hàng cung cấp, mọi nhu cầu liên quan đến việc nâng cấp, thay thế linh kiện, sửa chữa... đều phải thông báo và nhận được sự cho phép của Khách hàng trước khi thực hiện.<br>- Trường hợp mang thiết bị ra khỏi phạm vi công ty (bao gồm cả trường hợp đi công tác), nhân viên có trách nhiệm hoàn tất thủ tục Request - Mang thiết bị ra bên ngoài phạm vi công ty và tuân thủ các nội dung trong Cam kết bảo mật khi mang thiết bị ra bên ngoài phạm vi công ty <br>+ Thời hạn sử dụng request: <br>          . Staff: Tối đa 3 tháng<br>          . PL/LM: Tối đa 6 tháng<br>          . Head/SDM/UM: Vô thời hạn<br>+ Kết thúc thời hạn sử dụng, nhân viên không được phép tiếp tục mang thiết bị ra ngoài phạm vi công ty. Trường hợp muốn tiếp tục mang thiết bị ra ngoài phạm vi công ty, nhân viên cần tạo lại Request trước ngày hết hạn.<br>- Khi thay đổi thiết bị hoặc mang thêm thiết bị khác ra khỏi phạm vi công ty (bao gồm cả trường hợp đi công tác) cần tạo Request mới và ký lại Cam kết bảo mật](https://asset.sun-asterisk.vn/my_requests)|[- Mọi trường hợp cần sử dụng thiết bị Cá nhân thay thế thiết bị do Công ty cung cấp cho mục đích công việc, đều phải được PL/LM đánh giá, phê duyệt với lý do sử dụng phù hợp.<br>- PL/LM có trách nhiệm xác nhận mục đích sử dụng thiết bị Cá nhân phù hợp với nhu cầu của công việc trước khi phê duyệt Request.<br>- Nhân viên có trách nhiệm hoàn tất thủ tục Request_Sử dụng thiết bị cá nhân trong công việc và tuân thủ các nội dung trong Cam kết bảo mật trong quá trình sử dụng thiết bị cá nhân trong suốt quá trình sử dụng thiết bị phục vụ cho công việc.<br>+ Thời hạn sử dụng request:<br>. Staff: Tối đa 3 tháng<br>. PL/LM: Tối đa 6 tháng<br>. Head/SDM/UM: Vô thời hạn<br>+ Kết thúc thời hạn sử dụng, nhân viên không được phép tiếp tục sử dụng thiết bị Cá nhân cho mục đích công việc. Trường hợp muốn tiếp tục sử dụng, nhân viên tiến hành tạo lại Request trước ngày hết hạn.<br>- Khi thay đổi thiết bị hoặc sử dụng thêm thiết bị cá nhân khác cần tạo Request mới và ký lại Cam kết bảo mật.](https://docs.google.com/forms/d/e/1FAIpQLSdnUOIuClgXSDzZpGXzog7LF_fsZ-7grV_GzRyiLXX3O8ejyA/viewform)|[- Mọi trường hợp cần sử dụng thiết bị Cá nhân thay thế thiết bị do Công ty cung cấp cho mục đích công việc, đều phải được PL/LM đánh giá, phê duyệt với lý do sử dụng phù hợp.<br>- PL/LM có trách nhiệm xác nhận mục đích sử dụng thiết bị Cá nhân phù hợp với nhu cầu của công việc trước khi phê duyệt Request.<br>- Nhân viên có trách nhiệm hoàn tất thủ tục Request_Sử dụng thiết bị cá nhân trong công việc và tuân thủ các nội dung trong Cam kết bảo mật trong quá trình sử dụng thiết bị cá nhân trong suốt quá trình sử dụng thiết bị phục vụ cho công việc.<br>+ Thời hạn sử dụng request:<br>. Staff: Tối đa 3 tháng<br>. PL/LM: Tối đa 6 tháng<br>. Head/SDM/UM: Vô thời hạn<br>+ Kết thúc thời hạn sử dụng, nhân viên không được phép tiếp tục sử dụng thiết bị Cá nhân cho mục đích công việc. Trường hợp muốn tiếp tục sử dụng, nhân viên tiến hành tạo lại Request trước ngày hết hạn.<br>- Khi thay đổi thiết bị hoặc sử dụng thêm thiết bị cá nhân khác cần tạo Request mới và ký lại Cam kết bảo mật.](https://docs.google.com/forms/d/e/1FAIpQLSdnUOIuClgXSDzZpGXzog7LF_fsZ-7grV_GzRyiLXX3O8ejyA/viewform)|
|**1. Quản lý thiết bị**|**1.2 Cho mượn/ sử dụng chung thiết bị**|[- Nghiêm cấm việc tự ý cho mượn/ sử dụng chung thiết bị do Công ty hoặc Khách hàng cung cấp<br>- Trường hợp các dự án khác nhau nhưng cùng chung một khách hàng, khi có nhu cầu mượn/ sử dụng chung thiết bị do khách hàng cung cấp cần nhận được sự cho phép của khách hàng.<br>- Với các thiết bị test sử dụng trong dự án:<br>+ Nghiêm cấm việc tự ý cho mượn thiết bị test giữa các dự án khác nhau, mọi nhu cầu thay đổi cần có sự cho phép của PL<br>  . Trường hợp cho mượn >5 ngày làm việc, dự án cần thực hiện thủ tục giao nhận thông qua hệ thống S*Asset và cần mang thiết bị test qua GA để tiến hành kiểm tra/cài lại máy trước khi cho mượn thiết bị<br>  . Trường hợp cho mượn <5 ngày làm việc, nhân viên đảm bảo logout tất cả các tài khoản trên các hệ thống, ứng dụng, dịch vụ và xóa/backup (nếu cần) các dữ liệu liên quan đến dự án trên thiết bị trước cho mượn thiết bị<br>+ Tất cả các trường hợp thay đổi thông tin người giữ thiết bị  khi phát sinh cho mượn thiết bị cần được cập nhật vào [Temp] [###(ProjectID)] ProjectName - Device Management](https://docs.google.com/spreadsheets/d/1nq-jTnhqV4f5yvssiOP-HEeuXivl_EG2Qwnr2od6MJE/edit)|[- Nghiêm cấm việc tự ý cho mượn/ sử dụng chung thiết bị do Công ty hoặc Khách hàng cung cấp<br>- Trường hợp các dự án khác nhau nhưng cùng chung một khách hàng, khi có nhu cầu mượn/ sử dụng chung thiết bị do khách hàng cung cấp cần nhận được sự cho phép của khách hàng.<br>- Với các thiết bị test sử dụng trong dự án:<br>+ Nghiêm cấm việc tự ý cho mượn thiết bị test giữa các dự án khác nhau, mọi nhu cầu thay đổi cần có sự cho phép của PL<br>  . Trường hợp cho mượn >5 ngày làm việc, dự án cần thực hiện thủ tục giao nhận thông qua hệ thống S*Asset và cần mang thiết bị test qua GA để tiến hành kiểm tra/cài lại máy trước khi cho mượn thiết bị<br>  . Trường hợp cho mượn <5 ngày làm việc, nhân viên đảm bảo logout tất cả các tài khoản trên các hệ thống, ứng dụng, dịch vụ và xóa/backup (nếu cần) các dữ liệu liên quan đến dự án trên thiết bị trước cho mượn thiết bị<br>+ Tất cả các trường hợp thay đổi thông tin người giữ thiết bị  khi phát sinh cho mượn thiết bị cần được cập nhật vào [Temp] [###(ProjectID)] ProjectName - Device Management](https://docs.google.com/spreadsheets/d/1nq-jTnhqV4f5yvssiOP-HEeuXivl_EG2Qwnr2od6MJE/edit)|- Đảm bảo logout tài khoản User Profile/ Use account dành cho công việc và tất cả các tài khoản trên các hệ thống, ứng dụng, dịch vụ được sử dụng cho mục đích công việc trước khi cho mượn/ sử dụng chung, nhằm hạn chế rủi ro lộ thông tin công việc, thao tác nhầm trên các hệ thống công cụ liên quan đến công việc trong.|- Đảm bảo logout tài khoản User Profile/ Use account dành cho công việc và tất cả các tài khoản trên các hệ thống, ứng dụng, dịch vụ được sử dụng cho mục đích công việc trước khi cho mượn/ sử dụng chung, nhằm hạn chế rủi ro lộ thông tin công việc, thao tác nhầm trên các hệ thống công cụ liên quan đến công việc trong.|
|**2. Quản lý truy cập**|**2.1 Truy cập hệ thống từ xa (VPN)**|[- Trường hợp cần truy cập hệ thống từ xa (VPN) cần thực hiện thủ tục<br>Request - Yêu cầu truy cập hệ thống từ xa (VPN), hoàn thành thủ tục ký  Cam kết bảo mật trong quá trình truy cập từ xa (VPN) và tuân thủ các nội dung trong Cam kết.<br>+ Thời hạn sử dụng request: <br>          . Staff: Tối đa 3 tháng<br>          . PL/LM: Tối đa 6 tháng<br>          . Head/SDM/UM: Vô thời hạn<br>+ Kết thúc thời hạn sử dụng, nhân viên không được phép tiếp tục truy cập hệ thống từ xa VPN. Trường hợp muốn tiếp tục truy cập hệ thống từ xa (VPN), nhân viên cầu tạo lại Request trước ngày hết hạn <br>- Không kết nối VPN cho các hoạt động ngoài mục đích công việc<br>- Không chia sẻ thông tin về tài khoản truy cập VPN cho người khác](https://docs.google.com/forms/d/e/1FAIpQLSfvEPINlVIQUcJIICcjxmfFH-ZyR2u2a4s-TvlZkjoYPN_wdg/viewform)|[- Trường hợp cần truy cập hệ thống từ xa (VPN) cần thực hiện thủ tục<br>Request - Yêu cầu truy cập hệ thống từ xa (VPN), hoàn thành thủ tục ký  Cam kết bảo mật trong quá trình truy cập từ xa (VPN) và tuân thủ các nội dung trong Cam kết.<br>+ Thời hạn sử dụng request: <br>          . Staff: Tối đa 3 tháng<br>          . PL/LM: Tối đa 6 tháng<br>          . Head/SDM/UM: Vô thời hạn<br>+ Kết thúc thời hạn sử dụng, nhân viên không được phép tiếp tục truy cập hệ thống từ xa VPN. Trường hợp muốn tiếp tục truy cập hệ thống từ xa (VPN), nhân viên cầu tạo lại Request trước ngày hết hạn <br>- Không kết nối VPN cho các hoạt động ngoài mục đích công việc<br>- Không chia sẻ thông tin về tài khoản truy cập VPN cho người khác](https://docs.google.com/forms/d/e/1FAIpQLSfvEPINlVIQUcJIICcjxmfFH-ZyR2u2a4s-TvlZkjoYPN_wdg/viewform)|[- Trường hợp cần truy cập hệ thống từ xa (VPN) cần thực hiện tuân thủ <br>Request - Yêu cầu truy cập hệ thống từ xa (VPN), hoàn thành thủ tục ký Cam kết bảo mật trong quá trình truy cập từ xa (VPN)  và tuân thủ các nội dung trong Cam kết.<br>+ Thời hạn sử dụng request: <br>          . Staff: Tối đa 3 tháng<br>          . PL/LM: Tối đa 6 tháng<br>          . Head/SDM/UM: Vô thời hạn<br>+ Kết thúc thời hạn sử dụng, nhân viên không được phép tiếp tục truy cập hệ thống từ xa VPN. Trường hợp muốn tiếp tục truy cập hệ thống từ xa (VPN), nhân viên cần tạo lại Request trước ngày hết hạn<br>- Đảm bảo sử dụng thông tin tài khoản đăng nhập truy cập từ xa VPN đúng mục đích, không chia sẻ thông tin về tài khoản truy cập VPN cho người khác<br>- Khi thực hiện các hoạt động ngoài mục đích công việc, cần bắt buộc ngắt kết nối VPN](https://docs.google.com/forms/d/e/1FAIpQLSfvEPINlVIQUcJIICcjxmfFH-ZyR2u2a4s-TvlZkjoYPN_wdg/viewform)|[- Trường hợp cần truy cập hệ thống từ xa (VPN) cần thực hiện tuân thủ <br>Request - Yêu cầu truy cập hệ thống từ xa (VPN), hoàn thành thủ tục ký Cam kết bảo mật trong quá trình truy cập từ xa (VPN)  và tuân thủ các nội dung trong Cam kết.<br>+ Thời hạn sử dụng request: <br>          . Staff: Tối đa 3 tháng<br>          . PL/LM: Tối đa 6 tháng<br>          . Head/SDM/UM: Vô thời hạn<br>+ Kết thúc thời hạn sử dụng, nhân viên không được phép tiếp tục truy cập hệ thống từ xa VPN. Trường hợp muốn tiếp tục truy cập hệ thống từ xa (VPN), nhân viên cần tạo lại Request trước ngày hết hạn<br>- Đảm bảo sử dụng thông tin tài khoản đăng nhập truy cập từ xa VPN đúng mục đích, không chia sẻ thông tin về tài khoản truy cập VPN cho người khác<br>- Khi thực hiện các hoạt động ngoài mục đích công việc, cần bắt buộc ngắt kết nối VPN](https://docs.google.com/forms/d/e/1FAIpQLSfvEPINlVIQUcJIICcjxmfFH-ZyR2u2a4s-TvlZkjoYPN_wdg/viewform)|
|**2. Quản lý truy cập**|**2.2 Truy cập mạng Wifi nội bộ**|- Tất cả thiết bị Công ty hoặc Khách hàng cung cấp đều được bộ phận Infra chủ động cập nhật địa chỉ MAC và cấp quyền truy cập vào Wifi Staff|- Tất cả thiết bị Công ty hoặc Khách hàng cung cấp đều được bộ phận Infra chủ động cập nhật địa chỉ MAC và cấp quyền truy cập vào Wifi Staff|[- Trường hợp cần truy cập vào mạng Wifi nội bộ (Sun-Staff), cần thực hiện thủ tục: <br>1. Request - Yêu cầu quyền truy cập wifi Staff cho thiết bị cá nhân  <br>2. Mang thiết bị qua bộ phận Infra để tiến hành format cài đặt lại máy theo quy định thiết bị của Công ty. <br>    + Nhân viên: Tự tiến hành sao lưu, lưu trữ dữ liệu cá nhân trước khi mang máy sang Infra để format. Và tự chịu trách nhiệm với những mất mát, thiệt hại do việc format máy gây ra<br>     + Bộ phận Infra: Trường hợp cần thực hiện xóa dữ liệu cá nhân phải được sự đồng ý và xác nhận của nhân viên<br>3. Nhân viên tuân thủ các nội dung của quy định ANTT và các điều khoản trong cam kết bảo mật <br>Cam kết bảo mật trong quá trình sử dụng thiết bị cá nhân và truy cập vào mạng nội bộ của công ty<br>](https://docs.google.com/forms/d/e/1FAIpQLSdnUOIuClgXSDzZpGXzog7LF_fsZ-7grV_GzRyiLXX3O8ejyA/viewform)|[- Trường hợp cần truy cập vào mạng Wifi nội bộ (Sun-Staff), cần thực hiện thủ tục: <br>1. Request - Yêu cầu quyền truy cập wifi Staff cho thiết bị cá nhân  <br>2. Nhân viên tuân thủ các nội dung của quy định ANTT và các điều khoản trong cam kết bảo mật <br>Cam kết bảo mật trong quá trình sử dụng thiết bị cá nhân và truy cập vào mạng nội bộ của công ty<br>](https://docs.google.com/forms/d/e/1FAIpQLSdnUOIuClgXSDzZpGXzog7LF_fsZ-7grV_GzRyiLXX3O8ejyA/viewform)|
|**2. Quản lý truy cập**|**2.3 Truy cập mạng Wifi công cộng**|- Không kết nối sử dụng Wi-Fi công cộng không biết rõ nguồn gốc, không có mật khẩu bảo vệ khi làm việc ngoài phạm vi công ty|- Không kết nối sử dụng Wi-Fi công cộng không biết rõ nguồn gốc, không có mật khẩu bảo vệ khi làm việc ngoài phạm vi công ty|Không kết nối sử dụng Wi-Fi công cộng không biết rõ nguồn gốc, không có mật khẩu bảo vệ khi làm việc ngoài phạm vi công ty|Không kết nối sử dụng Wi-Fi công cộng không biết rõ nguồn gốc, không có mật khẩu bảo vệ khi làm việc ngoài phạm vi công ty|
|**2. Quản lý truy cập**|**2.4 Kết nối không dây (Bluetooth, Airdrop...)**|- Đảm bảo xác minh thiết bị (đúng ID và tên thiết bị) trước khi ghép đôi <br>- Việc chia sẻ các thông tin có yêu cầu bảo mật cao cần có sự phê duyệt của PL/LM<br>- Đảm bảo tắt chức năng kết nối không dây khi không có nhu cầu sử dụng|- Đảm bảo xác minh thiết bị (đúng ID và tên thiết bị) trước khi ghép đôi <br>- Việc chia sẻ các thông tin có yêu cầu bảo mật cao cần có sự phê duyệt của PL/LM<br>- Đảm bảo tắt chức năng kết nối không dây khi không có nhu cầu sử dụng|- Đảm bảo xác minh thiết bị (đúng ID và tên thiết bị) trước khi ghép đôi<br>- Việc chia sẻ các thông tin có yêu cầu bảo mật cao cần có sự phê duyệt của PL/LM<br>- Đảm bảo luôn tắt chức năng kết nối không dây khi sử dụng thiết bị cho mục đích công việc.|- Đảm bảo xác minh thiết bị (đúng ID và tên thiết bị) trước khi ghép đôi<br>- Việc chia sẻ các thông tin có yêu cầu bảo mật cao cần có sự phê duyệt của PL/LM<br>- Đảm bảo luôn tắt chức năng kết nối không dây khi sử dụng thiết bị cho mục đích công việc.|
|**2. Quản lý truy cập**|**2.5 Truy cập internet an toàn**|- Không nhấp vào bất kỳ quảng cáo trên bất kỳ trang web hoặc bất kỳ tệp đính kèm nào được gửi qua thư hay SMS<br>- Không nhấp vào các đường link lạ (link độc), các đường link nhận được từ các địa chỉ lạ|- Không nhấp vào bất kỳ quảng cáo trên bất kỳ trang web hoặc bất kỳ tệp đính kèm nào được gửi qua thư hay SMS<br>- Không nhấp vào các đường link lạ (link độc), các đường link nhận được từ các địa chỉ lạ|- Không nhấp vào bất kỳ quảng cáo trên bất kỳ trang web hoặc bất kỳ tệp đính kèm nào được gửi qua thư hay SMS<br>- Không nhấp vào các đường link lạ (link độc), các đường link nhận được từ các địa chỉ lạ|- Không nhấp vào bất kỳ quảng cáo trên bất kỳ trang web hoặc bất kỳ tệp đính kèm nào được gửi qua thư hay SMS<br>- Không nhấp vào các đường link lạ (link độc), các đường link nhận được từ các địa chỉ lạ|
|**2. Quản lý truy cập**|**2.6 Truy cập môi trường Production (server) trong quá trình phát triển dự án**| | |- Nghiêm cấm sử dụng thiết bị cá nhân truy cập vào môi trường Production (bao gồm cả việc test dự án). Trừ trường hợp được SDM phê duyệt sử dụng để thay thế, bổ sung do thiết bị công ty không đáp ứng hoặc trong một số trường bất khả kháng khác (vd: khi thành viên không có sẵn thiết bị Công ty nhưng công việc cần hỗ trợ kịp thời...)|- Nghiêm cấm sử dụng thiết bị cá nhân truy cập vào môi trường Production (bao gồm cả việc test dự án). Trừ trường hợp được SDM phê duyệt sử dụng để thay thế, bổ sung do thiết bị công ty không đáp ứng hoặc trong một số trường bất khả kháng khác (vd: khi thành viên không có sẵn thiết bị Công ty nhưng công việc cần hỗ trợ kịp thời...)|
|**3. Thiết lập và cài đặt thiết bị**|**3.1 Hệ điều hành của thiết bị**|[- Nghiêm cấm việc tự ý cài lại, tự ý nâng cấp hoặc cài đặt các Hệ điều hành khác <br>- Mọi yêu cầu liên quan đến thay đổi OS cần phải thực hiện thông qua hệ thống S*Asset và cần có sự cho phép của PL/LM](https://asset.sun-asterisk.vn/my_requests)|[- Nghiêm cấm việc tự ý cài lại, tự ý nâng cấp hoặc cài đặt các Hệ điều hành khác <br>- Mọi yêu cầu liên quan đến thay đổi OS cần phải thực hiện thông qua hệ thống S*Asset và cần có sự cho phép của PL/LM](https://asset.sun-asterisk.vn/my_requests)| | |
|**3. Thiết lập và cài đặt thiết bị**|**3.2 Chế độ tắt máy**|- Tắt hoàn toàn (Shut down) sau khi kết thúc phiên làm việc.<br>- Duy trì chế độ tự động khóa màn hình sau khoảng thời gian không quá 5 phút khi không sử dụng<br>- Thực hiện khóa màn hình khi rời khỏi vị trí đặt thiết bị|- Duy trì chế độ tự động khóa màn hình sau khoảng thời gian không quá 1 phút khi không sử dụng<br>- Thực hiện khóa màn hình khi rời khỏi vị trí đặt thiết bị|- Duy trì chế độ tự động khóa màn hình sau khoảng thời gian không quá 5 phút khi không sử dụng<br>- Thực hiện khóa màn hình khi rời khỏi vị trí đặt thiết bị|- Duy trì chế độ tự động khóa màn hình sau khoảng thời gian không quá 1 phút khi không sử dụng<br>- Thực hiện khóa màn hình khi rời khỏi vị trí đặt thiết bị|
|**3. Thiết lập và cài đặt thiết bị**|**3.3 Tài khoản người dùng **|- Chỉ sử dụng tài khoản người dùng (User profile) là tài khoản với domain công ty trên thiết bị|- Chỉ sử dụng tài khoản người dùng (User profile) là tài khoản với domain công ty trên thiết bị|- Tách biệt tài khoản công việc và tài khoản cá nhân trên các trình duyệt web.<br>- Tách biệt tài khoản User profile/ Use account dùng cho mục đích công việc và cá nhân trên thiết bị|- Tách biệt tài khoản cá nhân và tài khoản dùng trong công việc.|
|**3. Thiết lập và cài đặt thiết bị**|**3.4 Mật khẩu khóa màn hình (Use Profile/ Use account)**|- Đảm bảo duy trì mật khẩu mạnh: Độ dài tối thiểu 8 ký tự, bao gồm ký tự viết hoa, ký tự đặc biệt, ký tự thường và số<br>- Định kỳ thay đổi mật khẩu 3 tháng/ lần.|Đảm bảo luôn duy trì sử dụng mật khẩu (mã Pin) theo quy định của công ty.|- Không dùng chung một mật khẩu tài khoản User Profile/ Use account dành cho công việc và cá nhân<br>- Đảm bảo duy trì mật khẩu mạnh: Độ dài tối thiểu 8 ký tự, bao gồm ký tự viết hoa, ký tự đặc biệt, ký tự thường và số<br>- Định kỳ thay đổi mật khẩu 3 tháng/ lần với User Profile/ Use account dành cho công việc.|Đảm bảo luôn duy trì cài đặt mật khẩu trên thiết bị.|
|**4. Cài đặt và sử dụng phần mềm**|**4.1 Cài đặt và sử dụng phần mềm**|[- Tuân thủ theo quy định: Mục A. Quy định ANTT - Phần II_Quy định về việc cài đặt, sử dụng dịch vụ phần mềm<br>- Nghiêm cấm tự ý cài đặt, sử dụng phần mềm trong danh mục phần mềm rủi ro cao Danh sách phần mềm rủi ro cao<br>- Không cài đặt, sử dụng các phần mềm không có bản quyền (bản crack), các ứng dụng lạ, không rõ nguồn gốc trên thiết bị cá nhân được dùng cho mục đích công việc nhằm hạn chế các rủi ro vi phạm bản quyền và rủi ro máy tính bị tấn công do nhiễm malware<br>- Kiểm tra nội dung các điều khoản sử dụng của gói phần mềm có đáp ứng phù hợp với mục đích làm việc hay không(Commercial use) (vd: Personal/Pro/Business...), nhằm hạn chế rủi ro vi phạm bản quyền cho quá trình làm việc<br>-  Không được phép sử dụng các license phần mềm do Công ty cung cấp cho mục đích cá nhân ](None)|[- Tuân thủ theo quy định: Mục A. Quy định ANTT - Phần II_Quy định về việc cài đặt, sử dụng dịch vụ phần mềm<br>- Nghiêm cấm tự ý cài đặt, sử dụng phần mềm trong danh mục phần mềm rủi ro cao Danh sách phần mềm rủi ro cao<br>- Không cài đặt, sử dụng các phần mềm không có bản quyền (bản crack), các ứng dụng lạ, không rõ nguồn gốc trên thiết bị cá nhân được dùng cho mục đích công việc nhằm hạn chế các rủi ro vi phạm bản quyền và rủi ro máy tính bị tấn công do nhiễm malware<br>- Kiểm tra nội dung các điều khoản sử dụng của gói phần mềm có đáp ứng phù hợp với mục đích làm việc hay không(Commercial use) (vd: Personal/Pro/Business...), nhằm hạn chế rủi ro vi phạm bản quyền cho quá trình làm việc<br>-  Không được phép sử dụng các license phần mềm do Công ty cung cấp cho mục đích cá nhân ](None)|- Không cài đặt, sử dụng các phần mềm không có bản quyền (bản crack), các ứng dụng lạ, không rõ nguồn gốc trên thiết bị cá nhân được dùng cho mục đích công việc nhằm hạn chế các rủi ro vi phạm bản quyền và rủi ro máy tính bị tấn công do nhiễm malware<br>- Kiểm tra nội dung các điều khoản sử dụng của gói phần mềm có đáp ứng phù hợp với mục đích làm việc hay không(Commercial use) (vd: Personal/Pro/Business...), nhằm hạn chế rủi ro vi phạm bản quyền cho quá trình làm việc<br>- Không được phép sử dụng các License phần mềm do Công ty cung cấp cho mục đích cá nhân<br>- Đảm bảo tất cả dữ liệu, hình ảnh liên quan đến công việc thiết bị cá nhân phải được xóa sau khi kết thúc quá trình làm việc/ kết thúc quá trình sử dụng thiết bị cá nhân|- Không cài đặt, sử dụng các phần mềm không có bản quyền (bản crack), các ứng dụng lạ, không rõ nguồn gốc trên thiết bị cá nhân được dùng cho mục đích công việc nhằm hạn chế các rủi ro vi phạm bản quyền và rủi ro máy tính bị tấn công do nhiễm malware<br>- Kiểm tra nội dung các điều khoản sử dụng của gói phần mềm có đáp ứng phù hợp với mục đích làm việc hay không(Commercial use) (vd: Personal/Pro/Business...), nhằm hạn chế rủi ro vi phạm bản quyền cho quá trình làm việc<br>- Không được phép sử dụng các License phần mềm do Công ty cung cấp cho mục đích cá nhân<br>- Đảm bảo tất cả dữ liệu, hình ảnh liên quan đến công việc thiết bị cá nhân phải được xóa sau khi kết thúc quá trình làm việc/ kết thúc quá trình sử dụng thiết bị cá nhân|
|**4. Cài đặt và sử dụng phần mềm**|**4.2 Thiếp lập an toàn**|[- Duy trì cài đặt phần mềm Antivirus (ESET, Avast, Kaspersky, ClamAV và Microsoft Defender...) và bật hệ thống Firewall được tích hợp trên các hệ điều hành (Windows, MAC, Ubuntu) trên các thiết bị<br>Hướng dẫn cài đặt phần mềm antivirus -> Hướng dẫn cài đặt phần mềm AV](https://docs.google.com/document/d/1RfSo4zGVDjp1CLZXQ0--StiRWeEDzSU1uwiLEk47VL8/edit)| |[- Duy trì bật hệ thống Firewall được tích hợp trong các hệ điều hành (Windows, MAC, Ubuntu) trên các thiết bị<br>- Cài đặt ít nhất một phần mềm antivirus uy tín, tuân thủ theo quy định bản quyền (ESET, Avast, Kaspersky, ClamAV và Microsoft Defender...)<br>Hướng dẫn cài đặt phần mềm antivirus -> Hướng dẫn cài đặt phần mềm AV](https://docs.google.com/document/d/1RfSo4zGVDjp1CLZXQ0--StiRWeEDzSU1uwiLEk47VL8/edit)| |
|**5. Xử lý sự cố**|**5.1 Xử lý khi mất, hỏng thiết bị hoặc khi xảy ra sự cố (tấn công mạng)**|[- Tuân thủ nội dung trong Quy định về việc sử dụng tài sản và xử lý mất hỏng tài sản công ty<br>[SHN] Quy định về việc sử dụng tài sản và xử lý mất, hỏng tài sản công ty<br>- Thông báo ngay với PL/LM phụ trách trực tiếp để và bộ phận ISO tiến hành đánh giá sự cố và đưa ra phương án xử lý. Trường hợp sự cố có khả năng tác động hoặc có tác động đến khách hàng cần thông báo ngay với Khách hàng<br>- Thông báo ngay với nhân viên phụ trách phòng Hạ tầng nội bộ (Infra) để hỗ trợ xử lý các tài khoản trên thiết bị và đối ứng với các phương án liên quan đến vấn đề kỹ thuật khác<br>- Chủ động thực hiện các phương pháp vô hiệu hóa, khóa thiết bị hoặc xóa dữ liệu từ xa (nếu có thể), chủ động thay đổi password đăng nhập trên các hệ thống, ứng dụng, dịch vụ ](https://drive.google.com/file/d/1miGuuWteg-HjIuzsHgzRflWmGVbjuPux/view?ts=65f7e2ab)|[- Tuân thủ nội dung trong Quy định về việc sử dụng tài sản và xử lý mất hỏng tài sản công ty<br>[SHN] Quy định về việc sử dụng tài sản và xử lý mất, hỏng tài sản công ty<br>- Thông báo ngay với PL/LM phụ trách trực tiếp để và bộ phận ISO tiến hành đánh giá sự cố và đưa ra phương án xử lý. Trường hợp sự cố có khả năng tác động hoặc có tác động đến khách hàng cần thông báo ngay với Khách hàng<br>- Thông báo ngay với nhân viên phụ trách phòng Hạ tầng nội bộ (Infra) để hỗ trợ xử lý các tài khoản trên thiết bị và đối ứng với các phương án liên quan đến vấn đề kỹ thuật khác<br>- Chủ động thực hiện các phương pháp vô hiệu hóa, khóa thiết bị hoặc xóa dữ liệu từ xa (nếu có thể), chủ động thay đổi password đăng nhập trên các hệ thống, ứng dụng, dịch vụ ](https://drive.google.com/file/d/1miGuuWteg-HjIuzsHgzRflWmGVbjuPux/view?ts=65f7e2ab)|- Thông báo ngay với PL/LM phụ trách trực tiếp và bộ phận ISO để tiến hành đánh giá sự cố và đưa ra phương án xử lý. Trường hợp sự cố có khả năng tác động hoặc có tác động đến khách hàng cần thông báo ngay với Khách hàng<br>- Thông báo ngay với phụ trách phòng Hạ tầng nội bộ (Infra) để hỗ trợ xử lý các tài khoản công việc trên thiết bị và đối ứng với các phương án liên quan đến vấn đề kỹ thuật khác liên quan đến tài khoản công việc hoặc tài liệu công việc<br>- Chủ động thực hiện các phương pháp chức năng vô hiệu hóa, khóa thiết bị hoặc xóa dữ liệu từ xa (nếu có thể), chủ động thay đổi password đăng nhập trên các hệ thống, ứng dụng, dịch vụ<br>Lưu ý: Thiết bị cá nhân là tài sản cá nhân thuộc sở hữu của nhân viên, nhân viên sẽ tự quản lý, sử dụng và tự chịu trách nhiệm với các hư hỏng, thiệt hại, mất mát đối với thiết bị đó|- Thông báo ngay với PL/LM phụ trách trực tiếp và bộ phận ISO để tiến hành đánh giá sự cố và đưa ra phương án xử lý. Trường hợp sự cố có khả năng tác động hoặc có tác động đến khách hàng cần thông báo ngay với Khách hàng<br>- Thông báo ngay với phụ trách phòng Hạ tầng nội bộ (Infra) để hỗ trợ xử lý các tài khoản công việc trên thiết bị và đối ứng với các phương án liên quan đến vấn đề kỹ thuật khác liên quan đến tài khoản công việc hoặc tài liệu công việc<br>- Chủ động thực hiện các phương pháp chức năng vô hiệu hóa, khóa thiết bị hoặc xóa dữ liệu từ xa (nếu có thể), chủ động thay đổi password đăng nhập trên các hệ thống, ứng dụng, dịch vụ<br>Lưu ý: Thiết bị cá nhân là tài sản cá nhân thuộc sở hữu của nhân viên, nhân viên sẽ tự quản lý, sử dụng và tự chịu trách nhiệm với các hư hỏng, thiệt hại, mất mát đối với thiết bị đó|
|**QUY ĐỊNH RIÊNG DÀNH CHO THIẾT BỊ TEST<br>Ngoài việc tuân thủ các Quy định chung, Thiết bị Test cần tuân thủ các Quy định riêng dưới đây:**|**QUY ĐỊNH RIÊNG DÀNH CHO THIẾT BỊ TEST<br>Ngoài việc tuân thủ các Quy định chung, Thiết bị Test cần tuân thủ các Quy định riêng dưới đây:**|**THIẾT BỊ CÔNG TY/ KHÁCH HÀNG**|**THIẾT BỊ CÔNG TY/ KHÁCH HÀNG**|**THIẾT BỊ CÁ NHÂN**|**THIẾT BỊ CÁ NHÂN**|
|[1. Sử dụng công cụ để trung chuyển dữ liệu, hình ảnh test](None)|[1. Sử dụng công cụ để trung chuyển dữ liệu, hình ảnh test](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|[2. Cài đặt và sử dụng phần mềm trên thiết bị Test](None)|[2. Cài đặt và sử dụng phần mềm trên thiết bị Test](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|[3. Thiết lập thiết bị Test](None)|[3. Thiết lập thiết bị Test](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|[4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)](None)|[4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)](None)|**LAPTOP/ PC**|**MOBILE/ TABLET**|**LAPTOP/ PC**|**MOBILE/ TABLET**|
|**1. Sử dụng công cụ để trung chuyển dữ liệu, hình ảnh test**|**1. Sử dụng công cụ để trung chuyển dữ liệu, hình ảnh test**|- Đối với thiết bị có hỗ trợ cài đặt phần mềm Nextcloud là các thiết bị có OS>13 trên IOS hoặc OS>6 trên Android dự án chỉ được sử dụng phần mềm Nextcloud để chuyển dữ liệu, hình ảnh test từ thiết bị test lên Laptop/PC |- Đối với thiết bị có hỗ trợ cài đặt phần mềm Nextcloud là các thiết bị có OS>13 trên IOS hoặc OS>6 trên Android dự án chỉ được sử dụng phần mềm Nextcloud để chuyển dữ liệu, hình ảnh test từ thiết bị test lên Laptop/PC |- Đối với thiết bị có hỗ trợ cài đặt phần mềm Nextcloud là các thiết bị có OS>13 trên IOS hoặc OS>6 trên Android dự án chỉ được sử dụng phần mềm Nextcloud để chuyển dữ liệu, hình ảnh test từ thiết bị test lên Laptop/PC |- Đối với thiết bị có hỗ trợ cài đặt phần mềm Nextcloud là các thiết bị có OS>13 trên IOS hoặc OS>6 trên Android dự án chỉ được sử dụng phần mềm Nextcloud để chuyển dữ liệu, hình ảnh test từ thiết bị test lên Laptop/PC |
|**1. Sử dụng công cụ để trung chuyển dữ liệu, hình ảnh test**|**1. Sử dụng công cụ để trung chuyển dữ liệu, hình ảnh test**|- Đối với các thiết bị test không hỗ trợ cài đặt phần mềm Nextcloud là các thiết bị có OS<13 trên IOS hoặc OS<6 trên Android và không hỗ trợ nâng cấp OS, dự án chủ động sử dụng các công cụ khác (Các công cụ lưu trữ và chia sẻ dữ liệu, hình ảnh được Khách hàng hoặc PL dự án phê duyệt). Và chỉ login bằng tài khoản mail với domain Công ty @sun-asterisk.com hoặc @sun-asterisk.vn do công ty cung cấp.  <br>- Đảm bảo logout tài khoản khi quá trình chuyển dữ liệu, hình ảnh test kết thúc nhằm hạn chế rủi ro rò rỉ thông tin dự án, dữ liệu cá nhân khi dùng chung thiết bị test|- Đối với các thiết bị test không hỗ trợ cài đặt phần mềm Nextcloud là các thiết bị có OS<13 trên IOS hoặc OS<6 trên Android và không hỗ trợ nâng cấp OS, dự án chủ động sử dụng các công cụ khác (Các công cụ lưu trữ và chia sẻ dữ liệu, hình ảnh được Khách hàng hoặc PL dự án phê duyệt). Và chỉ login bằng tài khoản mail với domain Công ty @sun-asterisk.com hoặc @sun-asterisk.vn do công ty cung cấp.  <br>- Đảm bảo logout tài khoản khi quá trình chuyển dữ liệu, hình ảnh test kết thúc nhằm hạn chế rủi ro rò rỉ thông tin dự án, dữ liệu cá nhân khi dùng chung thiết bị test|- Đối với các thiết bị test không hỗ trợ cài đặt phần mềm Nextcloud là các thiết bị có OS<13 trên IOS hoặc OS<6 trên Android và không hỗ trợ nâng cấp OS, dự án chủ động sử dụng các công cụ khác (Các công cụ lưu trữ và chia sẻ dữ liệu, hình ảnh test được Khách hàng hoặc PL dự án phê duyệt). Và chỉ login bằng tài khoản mail với domain Công ty @sun-asterisk.com hoặc @sun-asterisk.vn do công ty cung cấp. <br>- Đảm bảo logout tài khoản khi quá trình chuyển dữ liệu, hình ảnh test kết thúc nhằm hạn chế rủi ro rò rỉ thông tin dự án|- Đối với các thiết bị test không hỗ trợ cài đặt phần mềm Nextcloud là các thiết bị có OS<13 trên IOS hoặc OS<6 trên Android và không hỗ trợ nâng cấp OS, dự án chủ động sử dụng các công cụ khác (Các công cụ lưu trữ và chia sẻ dữ liệu, hình ảnh test được Khách hàng hoặc PL dự án phê duyệt). Và chỉ login bằng tài khoản mail với domain Công ty @sun-asterisk.com hoặc @sun-asterisk.vn do công ty cung cấp. <br>- Đảm bảo logout tài khoản khi quá trình chuyển dữ liệu, hình ảnh test kết thúc nhằm hạn chế rủi ro rò rỉ thông tin dự án|
|**1. Sử dụng công cụ để trung chuyển dữ liệu, hình ảnh test**|**1. Sử dụng công cụ để trung chuyển dữ liệu, hình ảnh test**|- Đảm bảo tất cả dữ liệu, hình ảnh test được xóa trên thiết bị test và trên các ứng dụng khi quá trình chuyển dữ liệu, hình ảnh test kết thúc|- Đảm bảo tất cả dữ liệu, hình ảnh test được xóa trên thiết bị test và trên các ứng dụng khi quá trình chuyển dữ liệu, hình ảnh test kết thúc|- Đảm bảo tất cả dữ liệu, hình ảnh test được xóa trên thiết bị test và trên các ứng dụng khi quá trình chuyển dữ liệu, hình ảnh test kết thúc|- Đảm bảo tất cả dữ liệu, hình ảnh test được xóa trên thiết bị test và trên các ứng dụng khi quá trình chuyển dữ liệu, hình ảnh test kết thúc|
|**2. Cài đặt và sử dụng phần mềm trên thiết bị Test**|**2. Cài đặt và sử dụng phần mềm trên thiết bị Test**|- Nghiêm cấm cài đặt các phần mềm không liên quan tới công việc|- Nghiêm cấm cài đặt các phần mềm không liên quan tới công việc| | |
|**2. Cài đặt và sử dụng phần mềm trên thiết bị Test**|**2. Cài đặt và sử dụng phần mềm trên thiết bị Test**|- Đối với thiết bị hỗ trợ cài đặt phần mềm Nextcloud, nghiêm cấm việc Login các ứng dụng tích hợp trong Google Workspace (Gmail, Google Docs, Sheets và Slides, Google Drive, Google Calendar, Google Meet, Google Forms, Google Chat) trên thiết bị test. Việc login tài khoản mail với domain công ty @sun-asterisk trên Google Workspace có thể tiềm ẩn rủi ro rò rỉ thông tin dự án, dữ liệu cá nhân khi dùng chung thiết bị test.|- Đối với thiết bị hỗ trợ cài đặt phần mềm Nextcloud, nghiêm cấm việc Login các ứng dụng tích hợp trong Google Workspace (Gmail, Google Docs, Sheets và Slides, Google Drive, Google Calendar, Google Meet, Google Forms, Google Chat) trên thiết bị test. Việc login tài khoản mail với domain công ty @sun-asterisk trên Google Workspace có thể tiềm ẩn rủi ro rò rỉ thông tin dự án, dữ liệu cá nhân khi dùng chung thiết bị test.| | |
|**2. Cài đặt và sử dụng phần mềm trên thiết bị Test**|**2. Cài đặt và sử dụng phần mềm trên thiết bị Test**|- Trường hợp do yêu cầu của dự án cần tải ứng dụng trên CH Play (Google Play) để phục vụ công việc, nhân viên đảm bảo logout tài khoản mail với domain Công ty @sun-asterisk.com trên thiết bị Android ngay sau khi hoàn tất quá trình tải xuống.|- Trường hợp do yêu cầu của dự án cần tải ứng dụng trên CH Play (Google Play) để phục vụ công việc, nhân viên đảm bảo logout tài khoản mail với domain Công ty @sun-asterisk.com trên thiết bị Android ngay sau khi hoàn tất quá trình tải xuống.| | |
|**3. Thiết lập thiết bị Test**|**3. Thiết lập thiết bị Test**|- Duy trì: <br>+ Tài khoản Icloud của Công ty trên thiết bị iOS;<br>+ Cài đặt On với chức năng Find My Iphone (bao gồm cả Find My network và Send Last Location)|- Duy trì: <br>+ Tài khoản Icloud của Công ty trên thiết bị iOS;<br>+ Cài đặt On với chức năng Find My Iphone (bao gồm cả Find My network và Send Last Location)| | |
|**3. Thiết lập thiết bị Test**|**3. Thiết lập thiết bị Test**|- Đảm bảo cài đặt Off với các chức năng: <br>+ Backup<br>+ Automatic restore<br>+ Syschronize (đồng bộ hóa)<br>+ Keychain<br>+ AutoFill Passwords<br>+ Share Location|- Đảm bảo cài đặt Off với các chức năng: <br>+ Backup<br>+ Automatic restore<br>+ Syschronize (đồng bộ hóa)<br>+ Keychain<br>+ AutoFill Passwords<br>+ Share Location|- Trong quá trình test, chú ý tắt tính năng đồng bộ hóa  tài khoản cá nhân, xoá tất cả các dữ liệu, hình ảnh test sau khi test xong và trước khi bật lại chức năng đồng bộ|- Trong quá trình test, chú ý tắt tính năng đồng bộ hóa  tài khoản cá nhân, xoá tất cả các dữ liệu, hình ảnh test sau khi test xong và trước khi bật lại chức năng đồng bộ|
|**3. Thiết lập thiết bị Test**|**3. Thiết lập thiết bị Test**|- Đảm bảo tất cả dữ liệu, hình ảnh test trên thiết bị test được xóa sau khi quá trình kiểm thử kết thúc hoặc trước khi cho mượn hay bàn giao thiết bị.|- Đảm bảo tất cả dữ liệu, hình ảnh test trên thiết bị test được xóa sau khi quá trình kiểm thử kết thúc hoặc trước khi cho mượn hay bàn giao thiết bị.|- Đảm bảo tất cả dữ liệu, hình ảnh test trên thiết bị phải được xóa sau khi quá trình kiểm thử kết thúc|- Đảm bảo tất cả dữ liệu, hình ảnh test trên thiết bị phải được xóa sau khi quá trình kiểm thử kết thúc|
|**4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)**|**4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)**|[- Đảm bảo tách biệt nhóm thiết bị được sử dụng cho hoạt động nghiên cứu bảo mật với thiết bị test sử dụng trong dự án. Khi dự án có nhu cầu kiểm thử, bộ phận nghiên cứu bảo mật không sử dụng nguồn thiết bị test của dự án mà chỉ sử dụng nguồn thiết bị phục vụ cho hoạt động nghiên cứu bảo mật<br>+ Người phụ trách thiết bị cập nhật Danh sách quản lý thiết bị phục vụ công việc nghiên cứu bảo mật theo biểu mẫu: [Temp] Test device management](https://docs.google.com/spreadsheets/d/1nq-jTnhqV4f5yvssiOP-HEeuXivl_EG2Qwnr2od6MJE/edit?gid=1797735761)|[- Đảm bảo tách biệt nhóm thiết bị được sử dụng cho hoạt động nghiên cứu bảo mật với thiết bị test sử dụng trong dự án. Khi dự án có nhu cầu kiểm thử, bộ phận nghiên cứu bảo mật không sử dụng nguồn thiết bị test của dự án mà chỉ sử dụng nguồn thiết bị phục vụ cho hoạt động nghiên cứu bảo mật<br>+ Người phụ trách thiết bị cập nhật Danh sách quản lý thiết bị phục vụ công việc nghiên cứu bảo mật theo biểu mẫu: [Temp] Test device management](https://docs.google.com/spreadsheets/d/1nq-jTnhqV4f5yvssiOP-HEeuXivl_EG2Qwnr2od6MJE/edit?gid=1797735761)|[- Đảm bảo tách biệt nhóm thiết bị được sử dụng cho hoạt động nghiên cứu bảo mật với thiết bị test sử dụng trong dự án. Khi dự án có nhu cầu kiểm thử, bộ phận nghiên cứu bảo mật không sử dụng nguồn thiết bị test của dự án mà chỉ sử dụng nguồn thiết bị phục vụ cho hoạt động nghiên cứu bảo mật<br>+ Người phụ trách thiết bị cập nhật Danh sách quản lý thiết bị phục vụ công việc nghiên cứu bảo mật theo biểu mẫu: [Temp] Test device management](https://docs.google.com/spreadsheets/d/1nq-jTnhqV4f5yvssiOP-HEeuXivl_EG2Qwnr2od6MJE/edit?gid=1797735761)|[- Đảm bảo tách biệt nhóm thiết bị được sử dụng cho hoạt động nghiên cứu bảo mật với thiết bị test sử dụng trong dự án. Khi dự án có nhu cầu kiểm thử, bộ phận nghiên cứu bảo mật không sử dụng nguồn thiết bị test của dự án mà chỉ sử dụng nguồn thiết bị phục vụ cho hoạt động nghiên cứu bảo mật<br>+ Người phụ trách thiết bị cập nhật Danh sách quản lý thiết bị phục vụ công việc nghiên cứu bảo mật theo biểu mẫu: [Temp] Test device management](https://docs.google.com/spreadsheets/d/1nq-jTnhqV4f5yvssiOP-HEeuXivl_EG2Qwnr2od6MJE/edit?gid=1797735761)|
|**4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)**|**4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)**|- Mọi tài khoản sử dụng trên thiết bị là tài khoản được tạo bởi Email có domain Công ty @sun-asterisk.com hoặc @sun-asterisk.vn do công ty cung cấp<br>+ Với thiết bị IOS, bộ phận chỉ sử dụng tài khoản Icloud được công ty cung cấp riêng để đăng nhập trên các thiết bị này. Bộ phận sử dụng thiết bị có trách nhiệm quản lý tất cả các thông tin tài khoản (ID, tên người dùng, mật khẩu, số điện thoại..).|- Mọi tài khoản sử dụng trên thiết bị là tài khoản được tạo bởi Email có domain Công ty @sun-asterisk.com hoặc @sun-asterisk.vn do công ty cung cấp<br>+ Với thiết bị IOS, bộ phận chỉ sử dụng tài khoản Icloud được công ty cung cấp riêng để đăng nhập trên các thiết bị này. Bộ phận sử dụng thiết bị có trách nhiệm quản lý tất cả các thông tin tài khoản (ID, tên người dùng, mật khẩu, số điện thoại..).|- Mọi tài khoản sử dụng trên thiết bị là tài khoản được tạo bởi Email có domain Công ty @sun-asterisk.com hoặc @sun-asterisk.vn do công ty cung cấp<br>+ Với thiết bị IOS, bộ phận chỉ sử dụng tài khoản Icloud được công ty cung cấp riêng để đăng nhập trên các thiết bị này. Bộ phận sử dụng thiết bị có trách nhiệm quản lý tất cả các thông tin tài khoản (ID, tên người dùng, mật khẩu, số điện thoại..).|- Mọi tài khoản sử dụng trên thiết bị là tài khoản được tạo bởi Email có domain Công ty @sun-asterisk.com hoặc @sun-asterisk.vn do công ty cung cấp<br>+ Với thiết bị IOS, bộ phận chỉ sử dụng tài khoản Icloud được công ty cung cấp riêng để đăng nhập trên các thiết bị này. Bộ phận sử dụng thiết bị có trách nhiệm quản lý tất cả các thông tin tài khoản (ID, tên người dùng, mật khẩu, số điện thoại..).|
|**4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)**|**4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)**|- Đảm bảo tuân thủ quy định về việc cài đặt sử dụng phần mềm bản quyền trên các thiết bị|- Đảm bảo tuân thủ quy định về việc cài đặt sử dụng phần mềm bản quyền trên các thiết bị|- Đảm bảo tuân thủ quy định về việc cài đặt sử dụng phần mềm bản quyền trên các thiết bị|- Đảm bảo tuân thủ quy định về việc cài đặt sử dụng phần mềm bản quyền trên các thiết bị|
|**4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)**|**4. Quy định sử dụng thiết bị phục vụ hoạt động nghiên cứu bảo mật (jailbreak/crack,...)**|- Bộ phận sử dụng thiết bị thiết bị có trách nhiệm theo dõi và đánh giá rủi ro định kỳ 6 tháng/lần|- Bộ phận sử dụng thiết bị thiết bị có trách nhiệm theo dõi và đánh giá rủi ro định kỳ 6 tháng/lần|- Bộ phận sử dụng thiết bị thiết bị có trách nhiệm theo dõi và đánh giá rủi ro định kỳ 6 tháng/lần|- Bộ phận sử dụng thiết bị thiết bị có trách nhiệm theo dõi và đánh giá rủi ro định kỳ 6 tháng/lần|
|**5. Cập nhật danh sách quản lý thiết bị**|**5. Cập nhật danh sách quản lý thiết bị**|[- Tất cả thông tin về thiết bị Test và những thay đổi trong quá trình sử dụng phải được cập nhật liên tục vào Danh sách quản lý thiết bị theo biểu mẫu:<br>[Temp] [###(ProjectID)] ProjectName - Device Management](https://docs.google.com/spreadsheets/d/1nq-jTnhqV4f5yvssiOP-HEeuXivl_EG2Qwnr2od6MJE/edit)|[- Tất cả thông tin về thiết bị Test và những thay đổi trong quá trình sử dụng phải được cập nhật liên tục vào Danh sách quản lý thiết bị theo biểu mẫu:<br>[Temp] [###(ProjectID)] ProjectName - Device Management](https://docs.google.com/spreadsheets/d/1nq-jTnhqV4f5yvssiOP-HEeuXivl_EG2Qwnr2od6MJE/edit)|[- PL/LM hoặc người phụ trách duy trì cập nhật các thông tin định kì ít nhất 1 tháng/ 1 lần về các thiết bị cá nhân được sử dụng cho mục đích Test, mục đích tiếp nhận các thông tin authentication các tài khoản dùng chung trên các service do Khách hàng cung cấp, hoặc các trường hợp khác mà PL/LM xác định cần thực hiện việc ghi nhận, quản lý thông tin của thiết bị cá nhân được sử dụng vì một mục đích chung của hoạt động dự án/ bộ phận, theo biểu mẫu: <br>+ Non_project: BM-1-501-01_Personal device management<br>+ Project: [Temp] [###(ProjectID)] ProjectName - Device Management](None)|[- PL/LM hoặc người phụ trách duy trì cập nhật các thông tin định kì ít nhất 1 tháng/ 1 lần về các thiết bị cá nhân được sử dụng cho mục đích Test, mục đích tiếp nhận các thông tin authentication các tài khoản dùng chung trên các service do Khách hàng cung cấp, hoặc các trường hợp khác mà PL/LM xác định cần thực hiện việc ghi nhận, quản lý thông tin của thiết bị cá nhân được sử dụng vì một mục đích chung của hoạt động dự án/ bộ phận, theo biểu mẫu: <br>+ Non_project: BM-1-501-01_Personal device management<br>+ Project: [Temp] [###(ProjectID)] ProjectName - Device Management](None)|
|[Lưu ý: Trong quá trình sử dụng thiết bị Test cần tuân thủ Quy định trong quá trình kiểm thử](https://docs.google.com/spreadsheets/d/133q7oZQASz9LIz5O8TGmCmwvz7xp731RkskpLUP6W1I/edit?gid=828988684)|[Lưu ý: Trong quá trình sử dụng thiết bị Test cần tuân thủ Quy định trong quá trình kiểm thử](https://docs.google.com/spreadsheets/d/133q7oZQASz9LIz5O8TGmCmwvz7xp731RkskpLUP6W1I/edit?gid=828988684)|[Lưu ý: Trong quá trình sử dụng thiết bị Test cần tuân thủ Quy định trong quá trình kiểm thử](https://docs.google.com/spreadsheets/d/133q7oZQASz9LIz5O8TGmCmwvz7xp731RkskpLUP6W1I/edit?gid=828988684)|[Lưu ý: Trong quá trình sử dụng thiết bị Test cần tuân thủ Quy định trong quá trình kiểm thử](https://docs.google.com/spreadsheets/d/133q7oZQASz9LIz5O8TGmCmwvz7xp731RkskpLUP6W1I/edit?gid=828988684)|[Lưu ý: Trong quá trình sử dụng thiết bị Test cần tuân thủ Quy định trong quá trình kiểm thử](https://docs.google.com/spreadsheets/d/133q7oZQASz9LIz5O8TGmCmwvz7xp731RkskpLUP6W1I/edit?gid=828988684)|[Lưu ý: Trong quá trình sử dụng thiết bị Test cần tuân thủ Quy định trong quá trình kiểm thử](https://docs.google.com/spreadsheets/d/133q7oZQASz9LIz5O8TGmCmwvz7xp731RkskpLUP6W1I/edit?gid=828988684)|
"""

gb_context = ""
fname = "large_table_gleaning_5"

asyncio.run(run(chunk, gb_context, f"output_dir/{fname}", fname))
